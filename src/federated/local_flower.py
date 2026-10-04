"""Run the repository's encrypted Flower clients/server on one local host."""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import numpy as np

from src.federated.server_strategy import LSTMEncryptedFedAvgStrategy
from src.performance.energy import CodeCarbonRoundEnergyMeter
from src.performance.metrics_logger import MetricsLogger
from src.security.key_manager import encode_key
from src.security.lstm_encrypted_update import parameters_to_decrypted_weights
from src.ml.lstm_model import build_lstm_model


def run_local_flower(data, epochs, batch_size, rounds, key, experiment_id,
                     log_path="results/performance_metrics.csv", target_scaler=None):
    """Run all four frozen Flower clients and return final model/round records.

    The server and clients use Flower's gRPC transport over loopback. This is
    an actual Flower execution but remains a single-host network simulation.
    """
    import flwr as fl
    from src.clients.client import EncryptedLSTMFlowerClient

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if target_scaler is None:
        raise ValueError("A shared target scaler fitted on the complete training split is required")
    target_mean = float(target_scaler.mean_[0])
    target_scale = float(target_scaler.scale_[0])
    if target_scaler.n_samples_seen_ != len(data["train"][1]):
        raise ValueError("Shared target scaler must be fitted on all 350 training targets")
    if not np.isfinite(target_mean) or not np.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Shared target scaler parameters must be finite with positive scale")
    prior_key = os.environ.get("FL_AES_KEY")
    clients_root = Path("data/client_data")
    client_ids = tuple(f"client_{i:02d}" for i in range(1, 5))
    counts = []
    client_rows = []
    for client_id in client_ids:
        from src.clients.client import load_client_dataset
        features, targets = load_client_dataset(clients_root / client_id)
        counts.append(len(targets))
        client_rows.append(np.column_stack((features[:, 0, :], targets)))
    if counts != [88, 88, 88, 86] or sum(counts) != len(data["train"][1]):
        raise ValueError(f"Frozen Flower client partition mismatch: {counts}")
    observed = np.concatenate(client_rows, axis=0)
    expected = np.column_stack((data["train"][0][:, 0, :], data["train"][1]))
    observed_order = np.lexsort(observed.T[::-1])
    expected_order = np.lexsort(expected.T[::-1])
    if not np.array_equal(observed[observed_order], expected[expected_order]):
        raise ValueError("Flower client CSVs do not partition the frozen training split exactly")
    os.environ["FL_AES_KEY"] = encode_key(key)

    def restore_key():
        if prior_key is None:
            os.environ.pop("FL_AES_KEY", None)
        else:
            os.environ["FL_AES_KEY"] = prior_key

    class CapturingStrategy(LSTMEncryptedFedAvgStrategy):
        def __init__(self, *args, **kwargs):
            self.round_log = []
            self.final_weights = None
            super().__init__(*args, **kwargs)

        def aggregate_fit(self, server_round, results, failures):
            client_records = []
            for proxy, fit_res in results:
                client_metrics = fit_res.metrics or {}
                client_row = {
                    "round": server_round,
                    "client_id": client_metrics.get("client_id", proxy.cid),
                    "number_of_clients": len(results),
                    "number_of_samples": fit_res.num_examples,
                    "training_latency_ms": client_metrics.get("training_latency_ms"),
                    "serialization_latency_ms": client_metrics.get("serialization_latency_ms"),
                    "encryption_latency_ms": client_metrics.get("encryption_latency_ms"),
                    "decryption_latency_ms": client_metrics.get("decryption_latency_ms"),
                    "encrypted_update_size_bytes": client_metrics.get("encrypted_update_size_bytes"),
                    "target_scaler_mean": client_metrics.get("target_scaler_mean"),
                    "target_scaler_scale": client_metrics.get("target_scaler_scale"),
                }
                client_records.append(client_row)
                self.metrics_logger.log(client_row)
            params, metrics_out = super().aggregate_fit(server_round, results, failures)
            if params is None or len(results) != 4 or failures:
                raise RuntimeError(
                    f"Flower round {server_round} incomplete: {len(results)} clients, {len(failures)} failures"
                )
            weights = parameters_to_decrypted_weights(params, key)
            self.final_weights = [np.array(w, copy=True) for w in weights]
            model = build_lstm_model(38)
            model.set_weights(weights)
            val_pred_scaled = model.predict(data["val"][0], verbose=0).reshape(-1)
            val_pred = val_pred_scaled * target_scale + target_mean
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            logged_rounds = []
            if self.metrics_logger.path.exists():
                import csv
                with self.metrics_logger.path.open(newline="", encoding="utf-8") as handle:
                    logged_rounds = [r for r in csv.DictReader(handle)
                                     if r.get("experiment_id") == self.experiment_id
                                     and r.get("round") == str(server_round)
                                     and not r.get("client_id")]
            round_latency_s = (float(logged_rounds[-1]["total_round_latency_ms"]) / 1000.0
                               if logged_rounds and logged_rounds[-1].get("total_round_latency_ms") else None)
            self.round_log.append({
                "round": server_round, "clients": len(results),
                "samples": sum(r.num_examples for _, r in results),
                "round_latency_s": round_latency_s,
                "val_mae": float(mean_absolute_error(data["val"][1], val_pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(data["val"][1], val_pred))),
                "val_r2": float(r2_score(data["val"][1], val_pred)),
                "energy_j": logged_rounds[-1].get("energy_joules") if logged_rounds else None,
                "energy_status": logged_rounds[-1].get("energy_status") if logged_rounds else "unavailable",
                "energy_method": logged_rounds[-1].get("energy_method") if logged_rounds else "unavailable",
                "client_records": client_records,
            })
            return params, metrics_out

    strategy = CapturingStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=4,
        min_evaluate_clients=0,
        min_available_clients=4,
        accept_failures=False,
        metrics_logger=MetricsLogger(log_path, experiment_id=experiment_id),
        energy_meter=CodeCarbonRoundEnergyMeter(),
        on_fit_config_fn=lambda server_round: {"server_round": server_round, "seed": 42},
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"127.0.0.1:{port}"
    server_error = []

    def serve():
        try:
            fl.server.start_server(
                server_address=address,
                config=fl.server.ServerConfig(num_rounds=rounds),
                strategy=strategy,
            )
        except BaseException as exc:
            server_error.append(exc)

    server_thread = threading.Thread(target=serve, name="flower-server", daemon=True)
    server_thread.start()
    time.sleep(1.0)
    client_errors = []

    def connect(client_id):
        try:
            client = EncryptedLSTMFlowerClient(
                client_id=client_id, data_root=clients_root,
                epochs=epochs, batch_size=batch_size, key=key,
                target_mean=target_mean, target_scale=target_scale,
            )
            fl.client.start_client(server_address=address, client=client)
        except BaseException as exc:
            client_errors.append((client_id, exc))

    client_threads = [threading.Thread(target=connect, args=(client_id,),
                                       name=f"flower-{client_id}", daemon=True)
                     for client_id in client_ids]
    for thread in client_threads:
        thread.start()
    deadline = time.monotonic() + 6 * 60 * 60
    server_thread.join(timeout=max(0, deadline - time.monotonic()))
    if server_thread.is_alive():
        restore_key()
        raise TimeoutError(f"Flower server did not finish {rounds} rounds before timeout")
    for thread in client_threads:
        thread.join(timeout=30)
    if server_error:
        restore_key()
        raise RuntimeError("Flower server failed") from server_error[0]
    if client_errors:
        detail = ", ".join(f"{cid}: {type(err).__name__}: {err}" for cid, err in client_errors)
        restore_key()
        raise RuntimeError(f"Flower client failure(s): {detail}")
    if len(strategy.round_log) != rounds or strategy.final_weights is None:
        restore_key()
        raise RuntimeError(f"Flower completed {len(strategy.round_log)}/{rounds} rounds")
    model = build_lstm_model(38)
    model.set_weights(strategy.final_weights)
    restore_key()
    return model, strategy.round_log
