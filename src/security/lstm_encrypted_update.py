"""
Encrypted LSTM model-update bridge.

Pipeline:
LSTM weights
    -> NPZ serialization
    -> AES-256-GCM encryption
    -> Flower Parameters

Sample count is kept separately as FL metadata.
"""

import numpy as np

from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from src.security.encryption import (
    AES_KEY_SIZE,
    NONCE_SIZE,
    encrypt_model_update,
)
from src.security.decryption import decrypt_model_update

from src.federated.serialize import (
    serialize_model_update,
    deserialize_model_update,
)


def serialize_and_encrypt_weights(
    weights: list[np.ndarray],
    key: bytes,
) -> bytes:
    """Serialize LSTM weights and encrypt them."""
    serialized = serialize_model_update(weights)

    return encrypt_model_update(
        serialized,
        key,
    )


def decrypt_and_deserialize_weights(
    encrypted_data: bytes,
    key: bytes,
) -> list[np.ndarray]:
    """Decrypt and restore LSTM weight arrays."""
    if not isinstance(encrypted_data, bytes):
        raise TypeError(
            "Encrypted weights must be bytes."
        )

    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must be exactly 32 bytes."
        )

    if len(encrypted_data) <= NONCE_SIZE:
        raise ValueError(
            "Encrypted weights are too short."
        )

    serialized = decrypt_model_update(
        encrypted_data,
        key,
    )

    return deserialize_model_update(
        serialized
    )


def encrypted_weights_to_parameters(
    weights: list[np.ndarray],
    key: bytes,
) -> Parameters:
    """Serialize + encrypt LSTM weights for Flower."""
    encrypted = serialize_and_encrypt_weights(
        weights,
        key,
    )

    encrypted_array = np.frombuffer(
        encrypted,
        dtype=np.uint8,
    ).copy()

    return ndarrays_to_parameters(
        [encrypted_array]
    )


def parameters_to_decrypted_weights(
    parameters: Parameters,
    key: bytes,
) -> list[np.ndarray]:
    """Decrypt Flower Parameters and restore LSTM weights."""
    arrays = parameters_to_ndarrays(
        parameters
    )

    if not arrays:
        raise ValueError(
            "Encrypted LSTM parameters are empty."
        )

    encrypted_data = arrays[0].tobytes()

    return decrypt_and_deserialize_weights(
        encrypted_data,
        key,
    )