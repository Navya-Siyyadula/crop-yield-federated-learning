
"""
Custom Flower Strategy for encrypted Random Forest clients.

Server flow:

Client encrypted model
        ↓
AES-256 decryption
        ↓
Random Forest model
        ↓
Federated tree aggregation
        ↓
Global Random Forest
        ↓
AES-256 encryption
        ↓
Flower global parameters
"""

from typing import List, Tuple, Optional, Dict

import flwr as fl
from flwr.common import FitRes, Parameters

from flwr.server.client_proxy import ClientProxy

from src.security.encrypted_update import (
    parameters_to_encrypted_model,
    encrypted_model_to_parameters,
)

from src.security.key_manager import (
    load_key_from_environment,
)

from src.federated.fedavg_forest import (
    get_client_update,
    aggregate_federated_forest,
)


class RFTreeAggregationStrategy(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[
            Tuple[ClientProxy, FitRes]
        ],
        failures,
    ) -> Tuple[
        Optional[Parameters],
        Dict,
    ]:

        if not results:
            return None, {}

        # Load shared AES-256 key
        encryption_key = (
            load_key_from_environment()
        )

        client_updates = []

        print(
            f"\nServer: decrypting "
            f"{len(results)} client updates..."
        )

        for _, fit_res in results:

            # Rebuild Flower Parameters
            encrypted_parameters = Parameters(
                tensor_type="",
                tensors=fit_res.parameters.tensors,
            )

            # Decrypt and restore client Random Forest
            model = parameters_to_encrypted_model(
                encrypted_parameters,
                encryption_key,
            )

            print(
                "Server: client model "
                "decrypted successfully."
            )

            client_updates.append(
                get_client_update(
                    model,
                    fit_res.num_examples,
                )
            )

        # Federated Random Forest aggregation
        global_model = aggregate_federated_forest(
            client_updates,
            total_trees=400,
        )

        print(
            "Server: global Random Forest "
            "aggregation completed."
        )

        # Encrypt the aggregated global model
        encrypted_global_parameters = (
            encrypted_model_to_parameters(
                global_model,
                encryption_key,
            )
        )

        print(
            "Server: global model "
            "encrypted successfully."
        )

        metrics = {
            "num_clients": len(results),
            "total_samples": sum(
                update["n_samples"]
                for update in client_updates
            ),
        }

        return (
            encrypted_global_parameters,
            metrics,
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        if not results:
            return None, {}

        maes = [
            result.metrics["mae"]
            for _, result in results
        ]

        avg_mae = sum(maes) / len(maes)

        return (
            avg_mae,
            {
                "avg_mae": avg_mae,
            },
        )
