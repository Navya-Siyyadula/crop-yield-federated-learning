"""
Flower strategy for encrypted LSTM Federated Learning.
"""

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


class LSTMEncryptedFedAvgStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy for AES-256-GCM encrypted LSTM updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the initial global LSTM model.
        self.initial_model = build_lstm_model(input_features=38)

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
            )

            client_updates.append(
                (
                    weights,
                    fit_res.num_examples,
                )
            )

        # Sample-count weighted FedAvg.
        global_weights = fedavg(client_updates)

        # Encrypt the new global model before Flower distributes it.
        encrypted_global_parameters = encrypted_weights_to_parameters(
            global_weights,
            encryption_key,
        )

        total_samples = sum(
            sample_count
            for _, sample_count in client_updates
        )

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