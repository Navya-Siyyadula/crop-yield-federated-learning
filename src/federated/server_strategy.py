"""
Flower strategy for encrypted LSTM Federated Learning.
"""

from time import perf_counter
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitRes, Parameters

from src.federated.fedavg import fedavg
from src.ml.lstm_model import build_lstm_model
from src.security.key_manager import load_key_from_environment
from src.security.lstm_encrypted_update import (
    encrypted_weights_to_parameters,
    parameters_to_decrypted_weights,
)
from src.performance.energy import EnergyMeter, calculate_elei
from src.performance.latency import LatencyCollector
from src.performance.metrics_logger import MetricsLogger


class LSTMEncryptedFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy for AES-256-GCM encrypted LSTM updates."""

    def __init__(self, *args, **kwargs):
        self.metrics_logger = kwargs.pop("metrics_logger", None) or MetricsLogger()
        self.energy_meter = kwargs.pop("energy_meter", None) or EnergyMeter()
        self.energy_reference_joules = kwargs.pop("energy_reference_joules", None)
        self.latency_reference_ms = kwargs.pop("latency_reference_ms", None)
        self.elei_w_energy = kwargs.pop("w_energy", 0.5)
        self.elei_w_latency = kwargs.pop("w_latency", 0.5)
        self.reference_source = kwargs.pop("reference_source", None)
        if (self.energy_reference_joules is None) != (self.latency_reference_ms is None):
            raise ValueError("Both ELEI reference values must be supplied together")
        if self.energy_reference_joules is not None:
            calculate_elei(
                0.0,
                0.0,
                self.energy_reference_joules,
                self.latency_reference_ms,
                energy_status="estimated",
                w_energy=self.elei_w_energy,
                w_latency=self.elei_w_latency,
            )
        self.experiment_id = self.metrics_logger.experiment_id
        self._round_started_at: dict[int, float] = {}
        super().__init__(*args, **kwargs)

        # Create the initial global LSTM model.
        self.initial_model = build_lstm_model(input_features=38)

    def configure_fit(self, server_round, parameters, client_manager):
        # This boundary starts the round interaction. It includes client work
        # and transport until aggregate_fit receives the returned updates; it
        # is not presented as pure network latency.
        self._round_started_at[server_round] = perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def initialize_parameters(
        self,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> Optional[Parameters]:
        """Provide encrypted initial LSTM weights to Flower."""

        key = load_key_from_environment()

        initial_weights = self.initial_model.get_weights()

        return encrypted_weights_to_parameters(
            initial_weights,
            key,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[
            Tuple[
                fl.server.client_proxy.ClientProxy,
                FitRes,
            ]
        ],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:

        if not results:
            return None, {}

        encryption_key = load_key_from_environment()

        client_updates = []
        latency = LatencyCollector()

        print(f"\n--- Federated Learning Round {server_round} ---")
        print(f"Received updates from {len(results)} clients.")

        for _, fit_res in results:

            encrypted_parameters = Parameters(
                tensor_type=fit_res.parameters.tensor_type,
                tensors=fit_res.parameters.tensors,
            )

            weights = parameters_to_decrypted_weights(
                encrypted_parameters,
                encryption_key,
                latency=latency,
            )

            client_updates.append(
                (
                    weights,
                    fit_res.num_examples,
                )
            )

        # Sample-count weighted FedAvg.
        started = perf_counter()
        global_weights = fedavg(client_updates)
        latency.record("aggregation_ms", (perf_counter() - started) * 1000.0)

        # Encrypt the new global model before Flower distributes it.
        encrypted_global_parameters = encrypted_weights_to_parameters(
            global_weights,
            encryption_key,
            latency=latency,
        )
        security_overhead = (latency.get("encryption_ms") or 0.0) + (
            latency.get("decryption_ms") or 0.0
        )
        if security_overhead > 0:
            latency.record("security_overhead_ms", security_overhead)

        total_samples = sum(
            sample_count
            for _, sample_count in client_updates
        )

        round_started_at = self._round_started_at.pop(server_round, None)
        if round_started_at is not None:
            latency.record("total_round_ms", (perf_counter() - round_started_at) * 1000.0)

        energy = self.energy_meter.read()
        latency_ms = latency.get("total_round_ms")
        # References are deliberately absent until an explicit baseline or
        # caller-supplied reference is configured.
        if (
            self.energy_reference_joules is not None
            and latency_ms is not None
            and energy.energy_joules is not None
        ):
            elei = calculate_elei(
                energy.energy_joules,
                latency_ms,
                self.energy_reference_joules,
                self.latency_reference_ms,
                energy_status=energy.energy_status,
                w_energy=self.elei_w_energy,
                w_latency=self.elei_w_latency,
            )
        else:
            elei = None
        metrics_row = {
            "round": server_round,
            "number_of_clients": len(results),
            "number_of_samples": total_samples,
            "decryption_latency_ms": latency.get("decryption_ms"),
            "deserialization_latency_ms": latency.get("deserialization_ms"),
            "aggregation_latency_ms": latency.get("aggregation_ms"),
            "security_overhead_ms": latency.get("security_overhead_ms"),
            "encryption_latency_ms": latency.get("encryption_ms"),
            "serialization_latency_ms": latency.get("serialization_ms"),
            "total_round_latency_ms": latency.get("total_round_ms"),
            "communication_latency_ms": None,
            "training_latency_ms": None,
            "energy_joules": energy.energy_joules,
            "energy_status": energy.energy_status,
            "energy_method": energy.energy_method,
            "ELEI": elei,
            "energy_reference_joules": self.energy_reference_joules,
            "latency_reference_ms": self.latency_reference_ms,
            "reference_source": self.reference_source,
        }
        self.metrics_logger.log(metrics_row)

        metrics = {
            "num_clients": float(len(results)),
            "total_samples": float(total_samples),
        }

        print(
            f"Round {server_round} aggregation complete."
        )
        print(
            f"Total training samples: {total_samples}"
        )

        return encrypted_global_parameters, metrics
