"""Reproducible frozen-split experiments for centralized and simulated FL LSTMs."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import time
from uuid import uuid4
from pathlib import Path

import numpy as np
import pandas as pd

from src.security.encryption import encrypt_model_update, generate_key
from src.federated.serialize import serialize_model_update, deserialize_model_update
from src.federated.fedavg import fedavg
from src.evaluation.target_scaling import (
    fit_target_scaler,
    inverse_transform_targets,
    transform_targets,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
SEED = 42
CLIENT_SIZES = (88, 88, 88, 86)


def load_frozen_data(*, include_test=True):
    """Load frozen train/validation splits and defer test loading when requested."""
    data = {}
    splits = [("train", 350), ("val", 75)]
    if include_test:
        splits.append(("test", 75))
    for split, n in splits:
        x = pd.read_csv(DATA / f"X_{split}.csv").to_numpy(dtype=np.float32)
        ydf = pd.read_csv(DATA / f"y_{split}.csv")
        if x.shape != (n, 38) or ydf.shape != (n, 1):
            raise ValueError(f"Unexpected frozen {split} shapes: {x.shape}, {ydf.shape}")
        if ydf.columns[0] != "yield_kg_per_hectare":
            raise ValueError(f"Unexpected target column: {ydf.columns[0]}")
        data[split] = (x.reshape(n, 1, 38), ydf.iloc[:, 0].to_numpy(dtype=np.float32))
    return data


def predict_original_units(model, x, target_scaler=None):
    predictions = model.predict(x, verbose=0).reshape(-1)
    if target_scaler is not None:
        predictions = inverse_transform_targets(target_scaler, predictions)
    return predictions


def metrics(model, x, y, target_scaler=None):
    pred = predict_original_units(model, x, target_scaler)
    return metrics_from_predictions(y, pred)


def metrics_from_predictions(y, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
    }


def fit_centralized(x, y, xv, yv, epochs, batch_size, callbacks=None):
    from src.ml.lstm_model import build_lstm_model
    model = build_lstm_model(38)
    history = model.fit(x, y, validation_data=(xv, yv), epochs=epochs,
                        batch_size=batch_size, verbose=0, shuffle=False,
                        callbacks=callbacks)
    model.experiment_history = history.history
    return model


def train_federated(x, y, epochs, batch_size, rounds, key, test_data=None):
    """Logical four-client FedAvg; updates are serialized and AES-GCM protected."""
    from src.ml.lstm_model import build_lstm_model
    if rounds < 1:
        raise ValueError("rounds must be positive")
    bounds = np.cumsum((0,) + CLIENT_SIZES)
    if bounds[-1] != len(x):
        raise ValueError("Client partition does not cover the frozen training set")
    global_model = build_lstm_model(38)
    weights = global_model.get_weights()
    records = []
    for round_id in range(1, rounds + 1):
        started = time.perf_counter()
        updates = []
        crypto_bytes = 0
        client_records = []
        for i, n in enumerate(CLIENT_SIZES):
            model = build_lstm_model(38)
            model.set_weights(weights)
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            local_start = time.perf_counter()
            model.fit(x[lo:hi], y[lo:hi], epochs=epochs,
                      batch_size=batch_size, verbose=0, shuffle=True)
            local_s = time.perf_counter() - local_start
            serialization_started = time.perf_counter()
            payload = serialize_model_update(model.get_weights())
            serialization_s = time.perf_counter() - serialization_started
            encryption_started = time.perf_counter()
            encrypted = encrypt_model_update(payload, key)
            encryption_s = time.perf_counter() - encryption_started
            crypto_bytes += len(payload) + len(encrypted)
            from src.security.decryption import decrypt_model_update
            decryption_started = time.perf_counter()
            cleartext = decrypt_model_update(encrypted, key)
            decryption_s = time.perf_counter() - decryption_started
            recovered = deserialize_model_update(cleartext)
            client_records.append({"round": round_id, "client_id": f"client_{i+1:02d}",
                                  "samples": n, "training_latency_s": local_s,
                                  "serialization_s": serialization_s, "encryption_s": encryption_s,
                                  "decryption_s": decryption_s, "plaintext_bytes": len(payload),
                                  "ciphertext_bytes": len(encrypted)})
            updates.append((recovered, n))
        weights = fedavg(updates)
        global_model.set_weights(weights)
        row = {"round": round_id, "round_latency_s": time.perf_counter() - started,
               "encrypted_payload_bytes": crypto_bytes, "client_records": client_records}
        if test_data is not None:
            row.update(metrics(global_model, *test_data))
        records.append(row)
    return global_model, records


def run_candidate(data, epochs, batch_size, rounds, key, target_scaler):
    from src.federated.local_flower import run_local_flower

    experiment_id = uuid4().hex
    model, rounds_log = run_local_flower(
        data, epochs, batch_size, rounds, key,
        experiment_id=experiment_id,
        log_path=RESULTS / "final_fl_normalized_performance_metrics.csv",
        target_scaler=target_scaler,
    )
    measured_rounds = [r["round_latency_s"] for r in rounds_log if r["round_latency_s"] is not None]
    latency = sum(measured_rounds) if len(measured_rounds) == rounds else None
    energy_values = [float(r["energy_j"]) for r in rounds_log
                     if r.get("energy_j") not in (None, "")]
    has_energy = len(energy_values) == rounds and all(
        r.get("energy_status") == "estimated" for r in rounds_log
    )
    energy = sum(energy_values) if has_energy else None
    val_scores = metrics(model, *data["val"], target_scaler=target_scaler)
    return {"local_epochs": epochs, "batch_size": batch_size, "latency_s": latency,
            "energy_j": energy, "energy_status": "estimated" if has_energy else "unavailable",
            "energy_method": "sum of per-round CodeCarbon task estimates" if has_energy else "CodeCarbon round energy incomplete",
            "val_mae": val_scores["mae"], "val_rmse": val_scores["rmse"],
            "val_r2": val_scores["r2"], "rounds": rounds,
            "completed_rounds": len(rounds_log), "round_log": rounds_log, "_model": model,
            "experiment_id": experiment_id}


def measure_task_energy(function, task_name):
    """Run a callable with CodeCarbon task estimation; never infer missing joules."""
    tracker = None
    task_started = False
    error = None
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(save_to_file=False, log_level="error", measure_power_secs=1)
        tracker.start()
        tracker.start_task(task_name)
        task_started = True
    except Exception as exc:
        error = f"CodeCarbon could not start: {type(exc).__name__}"
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    energy = None
    status = "unavailable"
    method = error or "CodeCarbon returned no task energy"
    if tracker is not None:
        try:
            task_data = tracker.stop_task(task_name) if task_started else None
            tracker.stop()
            value = getattr(task_data, "energy_consumed", None)
            if value is not None and float(value) >= 0:
                energy = float(value) * 3_600_000.0
                status = "estimated"
                method = "CodeCarbon software estimate (whole training task, joules converted from kWh)"
        except Exception as exc:
            method = f"CodeCarbon estimate failed: {type(exc).__name__}"
    return result, elapsed, energy, status, method


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def nondominated_fronts(objectives):
    """Return NSGA-II non-dominated fronts for minimization objectives."""
    dominates_set = [[] for _ in objectives]
    dominated_by = [0] * len(objectives)
    fronts = [[]]
    for p, a in enumerate(objectives):
        for q, b in enumerate(objectives):
            if p == q:
                continue
            if dominates(a, b):
                dominates_set[p].append(q)
            elif dominates(b, a):
                dominated_by[p] += 1
        if dominated_by[p] == 0:
            fronts[0].append(p)
    index = 0
    while index < len(fronts) and fronts[index]:
        next_front = []
        for p in fronts[index]:
            for q in dominates_set[p]:
                dominated_by[q] -= 1
                if dominated_by[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        index += 1
    return fronts


def crowding_distance(front, objectives):
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for k in range(len(objectives[0])):
        ordered = sorted(front, key=lambda i: objectives[i][k])
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        low, high = objectives[ordered[0]][k], objectives[ordered[-1]][k]
        if high == low:
            continue
        for j in range(1, len(ordered) - 1):
            distance[ordered[j]] += (objectives[ordered[j + 1]][k] - objectives[ordered[j - 1]][k]) / (high - low)
    return distance


def run_experiment(rounds=20, central_epochs=20, population=6, generations=3):
    """Run comparison and measurement-backed NSGA-II over discrete controls."""
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)
    RESULTS.mkdir(exist_ok=True)
    data = load_frozen_data(include_test=False)
    key = generate_key()
    x, y = data["train"]
    xv, yv = data["val"]
    target_scaler = fit_target_scaler(y)
    y_train_scaled = transform_targets(target_scaler, y)
    y_val_scaled = transform_targets(target_scaler, yv)
    central_checkpoint = RESULTS / "final_centralized_best_validation.weights.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(central_checkpoint), monitor="val_loss", mode="min",
        save_best_only=True, save_weights_only=True, verbose=0,
    )
    centralized, central_time, central_energy, central_energy_status, central_energy_method = measure_task_energy(
        lambda: fit_centralized(
            x, y_train_scaled, xv, y_val_scaled, central_epochs, 32,
            callbacks=[checkpoint_callback],
        ), "centralized_lstm"
    )
    centralized.load_weights(central_checkpoint)
    # Logical edge-cloud simulation using actual Flower clients/server locally.
    from src.federated.local_flower import run_local_flower
    edge_model, edge_rounds = run_local_flower(
        data, central_epochs, 32, 1, key,
        experiment_id=uuid4().hex,
        log_path=RESULTS / "final_fl_normalized_performance_metrics.csv",
        target_scaler=target_scaler,
    )
    edge_time = edge_rounds[0]["round_latency_s"]
    edge_energy = edge_rounds[0]["energy_j"]
    edge_energy_status = edge_rounds[0]["energy_status"]

    configs = [(e, b) for e in range(1, 6) for b in (16, 32, 64)]
    measurements = []
    seen = set()

    def evaluate(individual, generation):
        config = tuple(individual)
        if config not in seen:
            row = run_candidate(data, config[0], config[1], rounds, key, target_scaler)
            row["generation"] = generation
            measurements.append(row)
            seen.add(config)
            _write_candidate_checkpoint(measurements)

    def fitness_rows(indices):
        return [(measurements[i]["latency_s"], measurements[i]["energy_j"])
                for i in indices]

    def select_indices(indices, count):
        valid = [i for i in indices if measurements[i]["energy_j"] is not None]
        if len(valid) < count:
            return sorted(indices, key=lambda i: (measurements[i]["latency_s"],
                                                   measurements[i]["local_epochs"],
                                                   measurements[i]["batch_size"]))[:count]
        objectives = fitness_rows(valid)
        fronts = nondominated_fronts(objectives)
        selected = []
        for front in fronts:
            mapped = [valid[j] for j in front]
            if len(selected) + len(mapped) <= count:
                selected.extend(mapped)
            else:
                dist = crowding_distance(front, objectives)
                selected.extend(valid[j] for j in sorted(front, key=lambda j: dist[j], reverse=True)
                                [:count - len(selected)])
                break
        return selected

    def tournament_parent(parent_ids):
        objectives = fitness_rows(parent_ids)
        fronts = nondominated_fronts(objectives)
        ranks, distances = {}, {}
        for rank, front in enumerate(fronts):
            for local_i in front:
                ranks[parent_ids[local_i]] = rank
            for local_i, distance in crowding_distance(front, objectives).items():
                distances[parent_ids[local_i]] = distance
        left, right = random.choices(parent_ids, k=2)
        if ranks[left] != ranks[right]:
            return left if ranks[left] < ranks[right] else right
        if distances[left] != distances[right]:
            return left if distances[left] > distances[right] else right
        return random.choice((left, right))

    # Genuine NSGA-II: measured objectives, binary tournament by rank/crowding,
    # discrete crossover/mutation, and elitist environmental selection.
    baseline_gene = (1, 32)
    population_genes = [baseline_gene] + random.sample(
        [candidate for candidate in configs if candidate != baseline_gene],
        min(max(population - 1, 0), len(configs) - 1),
    )
    for generation in range(max(1, generations)):
        for gene in population_genes:
            evaluate(gene, generation)
        current_configs = {tuple(gene) for gene in population_genes}
        if any(measurements[next(i for i, row in enumerate(measurements)
                                 if (row["local_epochs"], row["batch_size"]) == config)]["energy_j"] is None
               for config in current_configs):
            # Both objectives are required for meaningful NSGA-II selection.
            break
        available = list(range(len(measurements)))
        parent_ids = select_indices(available, min(population, len(available)))
        if generation + 1 == generations:
            break
        offspring = []
        genes = [(measurements[i]["local_epochs"], measurements[i]["batch_size"])
                 for i in parent_ids]
        parent_gene = {i: genes[j] for j, i in enumerate(parent_ids)}
        attempts = 0
        while len(offspring) < population and len(seen) + len(offspring) < len(configs):
            attempts += 1
            if attempts > 100:
                break
            a = parent_gene[tournament_parent(parent_ids)]
            b = parent_gene[tournament_parent(parent_ids)]
            child = [random.choice((a[0], b[0])), random.choice((a[1], b[1]))]
            if random.random() < 0.25:
                child[0] = random.randint(1, 5)
            if random.random() < 0.25:
                child[1] = random.choice((16, 32, 64))
            if tuple(child) not in seen and child not in offspring:
                offspring.append(tuple(child))
        if not offspring:
            break
        for gene in offspring:
            evaluate(gene, generation + 1)
        parent_ids = select_indices(list(range(len(measurements))),
                                    min(population, len(measurements)))
        population_genes = [(measurements[i]["local_epochs"], measurements[i]["batch_size"])
                            for i in parent_ids]

    usable_ids = [i for i, r in enumerate(measurements)
                  if r["energy_j"] is not None and r["latency_s"] is not None]
    pareto = set()
    if usable_ids:
        objective_vectors = fitness_rows(usable_ids)
        pareto = {usable_ids[i] for i in nondominated_fronts(objective_vectors)[0]}
    ideal_latency = min((measurements[i]["latency_s"] for i in pareto), default=None)
    ideal_energy = min((measurements[i]["energy_j"] for i in pareto), default=None)
    latency_span = (max((measurements[i]["latency_s"] for i in pareto), default=0.0)
                    - ideal_latency) if ideal_latency is not None else None
    energy_span = (max((measurements[i]["energy_j"] for i in pareto), default=0.0)
                   - ideal_energy) if ideal_energy is not None else None
    for i, row in enumerate(measurements):
        row["solution_id"] = f"candidate_{i+1:03d}"
        row["generation"] = row.get("generation", 0)
        row["pareto_optimal"] = i in pareto
        row["selection_status"] = "not selected"
        row["client_configuration"] = "client_01..client_04 (88,88,88,86)"
        row["emo_method"] = "NSGA-II" if pareto else "NSGA-II not evaluated: energy objective unavailable"
        row["objective_1"] = row["latency_s"]
        row["objective_2"] = row["energy_j"]
        row["n_objectives"] = 2
        row["normalized_latency"] = (
            (row["latency_s"] - ideal_latency) / (latency_span or 1.0)
            if i in pareto else None
        )
        row["normalized_energy"] = (
            (row["energy_j"] - ideal_energy) / (energy_span or 1.0)
            if i in pareto else None
        )
    chosen = None
    if pareto:
        chosen = min(pareto, key=lambda i: (
            measurements[i]["normalized_latency"] ** 2
            + measurements[i]["normalized_energy"] ** 2
        ))
        measurements[chosen]["selection_status"] = "selected: normalized Pareto knee (closest to ideal point)"

    baseline_index = next(
        i for i, result in enumerate(measurements)
        if (result["local_epochs"], result["batch_size"]) == baseline_gene
    )
    final_index = chosen if chosen is not None else baseline_index

    # Load held-out test targets only after NSGA-II selection is complete.
    test_x = pd.read_csv(DATA / "X_test.csv").to_numpy(dtype=np.float32)
    test_y_frame = pd.read_csv(DATA / "y_test.csv")
    if test_x.shape != (75, 38) or test_y_frame.shape != (75, 1):
        raise ValueError(f"Unexpected frozen test shapes: {test_x.shape}, {test_y_frame.shape}")
    if test_y_frame.columns[0] != "yield_kg_per_hectare":
        raise ValueError(f"Unexpected test target: {test_y_frame.columns[0]}")
    test_x = test_x.reshape(75, 1, 38)
    test_y = test_y_frame.iloc[:, 0].to_numpy(dtype=np.float32)

    baseline_model = measurements[baseline_index]["_model"]
    selected_model = measurements[final_index]["_model"]

    # Generate each held-out prediction once, after configuration selection.
    central_pred = predict_original_units(centralized, test_x, target_scaler)
    edge_pred = predict_original_units(edge_model, test_x, target_scaler)
    baseline_pred = predict_original_units(baseline_model, test_x, target_scaler)
    selected_pred = baseline_pred if final_index == baseline_index else predict_original_units(
        selected_model, test_x, target_scaler
    )
    central_metrics = metrics_from_predictions(test_y, central_pred)
    edge_metrics = metrics_from_predictions(test_y, edge_pred)
    baseline_metrics = metrics_from_predictions(test_y, baseline_pred)
    selected_metrics = baseline_metrics if final_index == baseline_index else metrics_from_predictions(
        test_y, selected_pred
    )

    # Independent sklearn recomputation from saved in-memory original-unit predictions.
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    independent_selected_metrics = {
        "mae": float(mean_absolute_error(test_y, selected_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, selected_pred))),
        "r2": float(r2_score(test_y, selected_pred)),
    }
    if not all(np.isclose(selected_metrics[k], independent_selected_metrics[k])
               for k in ("mae", "rmse", "r2")):
        raise RuntimeError("Independent test metric verification failed")

    rows = [
        {"approach": "Centralized LSTM", **central_metrics, "training_time_s": central_time,
         "latency_s": central_time, "energy_j": central_energy, "energy_status": central_energy_status,
         "energy_method": central_energy_method,
         "rounds": 0, "clients": 1, "local_epochs": central_epochs, "batch_size": 32,
         "security": "N/A", "pareto_solution": "N/A", "emo_method": "N/A", "nsga_algorithm": "N/A", "objective_1": "N/A", "objective_2": "N/A"},
        {"approach": "Edge-Cloud LSTM (logical simulation)", **edge_metrics, "training_time_s": edge_time,
         "latency_s": edge_time, "energy_j": edge_energy, "energy_status": edge_energy_status,
         "energy_method": edge_rounds[0].get("energy_method", "CodeCarbon per-round estimate"),
         "rounds": 1, "clients": 4, "local_epochs": central_epochs, "batch_size": 32,
         "security": "AES-256-GCM updates", "pareto_solution": "N/A", "emo_method": "N/A", "nsga_algorithm": "N/A", "objective_1": "N/A", "objective_2": "N/A"},
    ]
    baseline_row = measurements[baseline_index]
    rows.append({"approach": "Federated LSTM", **baseline_metrics,
                 "training_time_s": baseline_row["latency_s"], "latency_s": baseline_row["latency_s"],
                 "energy_j": baseline_row["energy_j"], "energy_status": baseline_row["energy_status"],
                 "energy_method": baseline_row["energy_method"], "rounds": baseline_row["completed_rounds"],
                 "clients": 4, "local_epochs": baseline_row["local_epochs"], "batch_size": baseline_row["batch_size"],
                 "security": "AES-256-GCM", "pareto_solution": "N/A",
                 "emo_method": "NSGA-II candidate; fixed baseline configuration",
                 "nsga_algorithm": "NSGA-II" if pareto else "Unavailable: energy objective missing",
                 "objective_1": baseline_row["latency_s"], "objective_2": baseline_row["energy_j"]})
    if chosen is not None:
        selected_row = measurements[chosen]
        rows.append({"approach": "EMO-selected Federated LSTM", **selected_metrics,
                     "training_time_s": selected_row["latency_s"], "latency_s": selected_row["latency_s"],
                     "energy_j": selected_row["energy_j"], "energy_status": selected_row["energy_status"],
                     "energy_method": selected_row["energy_method"], "rounds": selected_row["completed_rounds"],
                     "clients": 4, "local_epochs": selected_row["local_epochs"], "batch_size": selected_row["batch_size"],
                     "security": "AES-256-GCM", "pareto_solution": selected_row["solution_id"],
                     "emo_method": "NSGA-II normalized Pareto knee selection",
                     "nsga_algorithm": "NSGA-II",
                     "objective_1": selected_row["latency_s"], "objective_2": selected_row["energy_j"]})

    _write_csv(RESULTS / "final_comparison.csv", rows)
    _write_csv(RESULTS / "emo_pareto_solutions.csv", measurements)
    round_records = []
    client_records = []
    for candidate in measurements:
        candidate_metadata = {
            "candidate_id": candidate["solution_id"],
            "experiment_id": candidate["experiment_id"],
            "local_epochs": candidate["local_epochs"],
            "batch_size": candidate["batch_size"],
        }
        for round_row in candidate["round_log"]:
            round_records.append({**candidate_metadata, **{
                key: value for key, value in round_row.items() if key != "client_records"
            }})
            client_records.extend({**candidate_metadata, **client}
                                  for client in round_row["client_records"])
    _write_csv(RESULTS / "fl_round_metrics.csv", round_records)
    _write_csv(RESULTS / "client_performance.csv", client_records)
    import importlib.metadata
    config = {"seed": SEED,
              "dataset_shapes": {**{s: list(v[0].shape) for s, v in data.items()},
                                 "test": list(test_x.shape)},
              "experiment_id": uuid4().hex,
              "target": "yield_kg_per_hectare", "clients": 4, "client_sizes": list(CLIENT_SIZES),
              "target_scaler": "StandardScaler",
              "target_scaler_fit_population": "all 350 frozen training targets across all clients",
              "target_scaler_fit_samples": int(target_scaler.n_samples_seen_),
              "target_scaler_mean": float(target_scaler.mean_[0]),
              "target_scaler_scale": float(target_scaler.scale_[0]),
              "all_clients_share_identical_target_scaler": True,
              "client_target_normalization": "(y_kg_per_hectare - target_scaler_mean) / target_scaler_scale",
              "global_prediction_evaluation": "inverse transform shared scaler then score in kg/ha",
              "centralized_checkpoint": "minimum validation loss, weights restored before test evaluation",
              "centralized_best_validation_epoch": int(np.argmin(centralized.experiment_history["val_loss"]) + 1),
              "centralized_best_validation_loss_standardized_mse": float(min(centralized.experiment_history["val_loss"])),
              "federated_client_checkpointing": "none; fixed candidate local epochs per Flower round",
              "rounds": rounds, "centralized_epochs": central_epochs,
              "fixed_federated_baseline_candidate": measurements[baseline_index]["solution_id"],
              "fixed_federated_baseline_local_epochs": measurements[baseline_index]["local_epochs"],
              "fixed_federated_baseline_batch_size": measurements[baseline_index]["batch_size"],
              "selected_candidate": measurements[chosen]["solution_id"] if chosen is not None else None,
              "selected_local_epochs": measurements[final_index]["local_epochs"],
              "selected_batch_size": measurements[final_index]["batch_size"],
              "selected_candidate_latency_s": measurements[final_index]["latency_s"],
              "selected_candidate_energy_j": measurements[final_index]["energy_j"],
              "selected_candidate_normalized_latency": measurements[final_index]["normalized_latency"],
              "selected_candidate_normalized_energy": measurements[final_index]["normalized_energy"],
              "pareto_ideal_latency_s": ideal_latency,
              "pareto_ideal_energy_j": ideal_energy,
              "local_epochs_bounds": [1, 5],
              "batch_size_choices": [16, 32, 64], "population": population,
              "generations": generations, "actual_candidates": len(measurements),
              "pareto_solution_count": len(pareto),
              "nsga_method": "NSGA-II non-dominated sorting and crowding distance",
              "selection_policy": "normalized Pareto knee (minimum distance to ideal point)",
              "energy_method": "CodeCarbon software estimate when energy telemetry is available; otherwise unavailable",
              "security": "AES-256-GCM", "python": platform.python_version(),
              "os": f"{platform.system()} {platform.release()}", "cpu": platform.processor(),
              "optimizer": "Adam", "learning_rate": 0.001,
              "tensorflow_intra_op_threads": 1, "tensorflow_inter_op_threads": 1,
              "tensorflow": __import__("tensorflow").__version__,
              "flower": importlib.metadata.version("flwr") if importlib.util.find_spec("flwr") else "unavailable",
              "fl_execution": "Flower server and four Flower clients over loopback; single-host simulation",
              "cryptography": importlib.metadata.version("cryptography"),
              "codecarbon": importlib.metadata.version("codecarbon") if importlib.util.find_spec("codecarbon") else "unavailable",
              "independent_selected_test_metrics_verified": True,
              "pymoo": "not used (NSGA-II implemented directly)"}
    (RESULTS / "final_experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_results(rows, measurements, chosen)
    return rows, measurements


def _write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row
                                 if key != "round_log" and key != "client_records"
                                 and not key.startswith("_")))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_checkpoint(measurements):
    """Persist completed, real candidate measurements during long searches."""
    _write_csv(RESULTS / "emo_candidate_checkpoint.csv", measurements)
    arrays = {}
    for index, row in enumerate(measurements):
        model = row.get("_model")
        if model is not None:
            arrays.update({f"candidate_{index:03d}_weight_{j:02d}": weight
                           for j, weight in enumerate(model.get_weights())})
    if arrays:
        np.savez_compressed(RESULTS / "emo_candidate_weights_checkpoint.npz", **arrays)


def _plot_results(comparison, candidates, chosen):
    """Create only plots supported by recorded measurements."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metric_specs = (("mae", "MAE", "model_vs_mae.png"),
                    ("rmse", "RMSE", "model_vs_rmse.png"),
                    ("r2", "R²", "model_vs_r2.png"),
                    ("latency_s", "Latency (s)", "model_vs_latency.png"))
    available = [r for r in comparison if isinstance(r.get("mae"), (float, int))]
    for metric, label, filename in metric_specs:
        if not available:
            continue
        fig, ax = plt.subplots()
        ax.bar([r["approach"] for r in available], [r[metric] for r in available])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(RESULTS / filename, dpi=160)
        plt.close(fig)
    valid = [(r["latency_s"], r["energy_j"], i) for i, r in enumerate(candidates)
             if r.get("energy_j") is not None]
    if valid:
        fig, ax = plt.subplots()
        ax.scatter([v[0] for v in valid], [v[1] for v in valid], label="Measured candidates")
        front = [v for v in valid if candidates[v[2]].get("pareto_optimal")]
        if front:
            ax.scatter([v[0] for v in front], [v[1] for v in front], label="Pareto optimal")
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Estimated energy (J)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "latency_vs_energy_pareto.png", dpi=160)
        plt.close(fig)
    energy_rows = [r for r in comparison if r.get("energy_j") not in (None, "N/A")]
    if energy_rows:
        fig, ax = plt.subplots()
        ax.bar([r["approach"] for r in energy_rows], [float(r["energy_j"]) for r in energy_rows])
        ax.set_ylabel("Estimated energy (J)")
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(RESULTS / "model_vs_energy.png", dpi=160)
        plt.close(fig)
    if chosen is not None:
        row = candidates[chosen]
        round_log = row.get("round_log", [])
        for key, label, filename in (("val_mae", "Validation MAE", "round_vs_mae.png"),
                                     ("val_rmse", "Validation RMSE", "round_vs_rmse.png"),
                                     ("round_latency_s", "Round latency (s)", "round_vs_latency.png")):
            if round_log and all(key in r for r in round_log):
                fig, ax = plt.subplots()
                ax.plot([r["round"] for r in round_log], [r[key] for r in round_log], marker="o")
                ax.set_xlabel("Communication round")
                ax.set_ylabel(label)
                fig.tight_layout()
                fig.savefig(RESULTS / filename, dpi=160)
                plt.close(fig)
        if round_log and all(r.get("energy_j") is not None for r in round_log):
            fig, ax = plt.subplots()
            ax.plot([r["round"] for r in round_log], [float(r["energy_j"]) for r in round_log], marker="o")
            ax.set_xlabel("Communication round")
            ax.set_ylabel("Estimated energy (J)")
            fig.tight_layout()
            fig.savefig(RESULTS / "round_vs_energy.png", dpi=160)
            plt.close(fig)
