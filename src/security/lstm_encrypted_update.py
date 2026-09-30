"""
Encrypted LSTM model-update bridge.

Pipeline:

LSTM weights
    -> NPZ serialization
    -> AES-256-GCM encryption
    -> Flower Parameters as raw bytes

The encrypted payload is treated as opaque bytes by Flower.
Sample count is kept separately as FL metadata.
"""

from time import perf_counter

import numpy as np

from flwr.common import Parameters

from src.security.encryption import AES_KEY_SIZE, NONCE_SIZE, encrypt_model_update
from src.security.decryption import decrypt_model_update
from src.federated.serialize import (
    serialize_model_update,
    deserialize_model_update,
)
from src.performance.latency import LatencyCollector


TENSOR_TYPE = "encrypted_lstm_aes256_gcm"


def serialize_and_encrypt_weights(
    weights: list[np.ndarray],
    key: bytes,
    *,
    latency: LatencyCollector | None = None,
) -> bytes:
    """Serialize LSTM weights and encrypt them with AES-256-GCM."""

    started = perf_counter()
    serialized = serialize_model_update(weights)
    if latency is not None:
        latency.record("serialization_ms", (perf_counter() - started) * 1000.0)

    started = perf_counter()
    encrypted = encrypt_model_update(
        serialized,
        key,
    )
    if latency is not None:
        latency.record("encryption_ms", (perf_counter() - started) * 1000.0)
    return encrypted


def decrypt_and_deserialize_weights(
    encrypted_data: bytes,
    key: bytes,
    *,
    latency: LatencyCollector | None = None,
) -> list[np.ndarray]:
    """Decrypt and deserialize an encrypted LSTM update."""

    if not isinstance(encrypted_data, bytes):
        raise TypeError("Encrypted weights must be bytes.")

    if len(key) != AES_KEY_SIZE:
        raise ValueError("AES-256 key must be exactly 32 bytes.")

    if len(encrypted_data) <= NONCE_SIZE:
        raise ValueError("Encrypted weights are too short.")

    started = perf_counter()
    serialized = decrypt_model_update(
        encrypted_data,
        key,
    )
    if latency is not None:
        latency.record("decryption_ms", (perf_counter() - started) * 1000.0)

    started = perf_counter()
    weights = deserialize_model_update(
        serialized,
    )
    if latency is not None:
        latency.record("deserialization_ms", (perf_counter() - started) * 1000.0)
    return weights


def encrypted_weights_to_parameters(
    weights: list[np.ndarray],
    key: bytes,
    *,
    latency: LatencyCollector | None = None,
) -> Parameters:
    """
    Convert encrypted LSTM weights into Flower Parameters.

    The encrypted payload is stored directly as bytes.
    It is NOT converted into a NumPy array because it is already
    encrypted opaque data.
    """

    encrypted = serialize_and_encrypt_weights(
        weights,
        key,
        latency=latency,
    )

    return Parameters(
        tensor_type=TENSOR_TYPE,
        tensors=[encrypted],
    )


def parameters_to_decrypted_weights(
    parameters: Parameters,
    key: bytes,
    *,
    latency: LatencyCollector | None = None,
) -> list[np.ndarray]:
    """
    Extract the raw encrypted payload from Flower Parameters,
    then decrypt and deserialize the LSTM weights.
    """

    if not isinstance(parameters, Parameters):
        raise TypeError("parameters must be a Flower Parameters object.")

    if not parameters.tensors:
        raise ValueError("Encrypted LSTM parameters are empty.")

    encrypted_data = parameters.tensors[0]

    if not isinstance(encrypted_data, bytes):
        raise TypeError("Encrypted LSTM parameter must be bytes.")

    return decrypt_and_deserialize_weights(
        encrypted_data,
        key,
        latency=latency,
    )
