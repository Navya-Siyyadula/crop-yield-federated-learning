"""
Encryption bridge for serialized federated learning model updates.

Flow:

Random Forest
    -> pickle serialization
    -> AES-256-GCM encryption
    -> decryption
    -> pickle deserialization
    -> Random Forest
"""
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import pickle

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.security.encryption import (
    AES_KEY_SIZE,
    NONCE_SIZE,
    generate_key,
    encrypt_model_update,
)



def serialize_model(model) -> bytes:
    """Serialize a model using the same pickle format as RF serialization."""

    return pickle.dumps(model)


def serialize_and_encrypt_model(model, key: bytes) -> bytes:
    """Serialize a trained model and encrypt the serialized bytes."""

    serialized_bytes = serialize_model(model)

    return encrypt_model_update(
        serialized_bytes,
        key,
    )


def decrypt_and_deserialize_model(
    encrypted_data: bytes,
    key: bytes,
):
    """Decrypt an encrypted model update and restore the model."""

    if not isinstance(encrypted_data, bytes):
        raise TypeError(
            "Encrypted model update must be bytes."
        )

    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must be exactly 32 bytes."
        )

    if len(encrypted_data) <= NONCE_SIZE:
        raise ValueError(
            "Encrypted model update is too short."
        )

    nonce = encrypted_data[:NONCE_SIZE]
    ciphertext = encrypted_data[NONCE_SIZE:]

    aes = AESGCM(key)

    serialized_bytes = aes.decrypt(
        nonce,
        ciphertext,
        None,
    )

    return pickle.loads(serialized_bytes)


def create_encryption_key() -> bytes:
    """Create a fresh AES-256 key for an FL session."""

    return generate_key()
def encrypted_model_to_parameters(model, key: bytes) -> Parameters:
    """
    Serialize and AES-encrypt a model, then wrap the
    encrypted bytes as Flower Parameters.
    """

    encrypted_data = serialize_and_encrypt_model(
        model,
        key,
    )

    encrypted_array = np.frombuffer(
        encrypted_data,
        dtype=np.uint8,
    ).copy()

    return ndarrays_to_parameters(
        [encrypted_array]
    )


def parameters_to_encrypted_model(
    parameters: Parameters,
    key: bytes,
):
    """
    Extract encrypted bytes from Flower Parameters,
    decrypt them, and restore the model.
    """

    arrays = parameters_to_ndarrays(
        parameters
    )

    if not arrays:
        raise ValueError(
            "Encrypted model parameters are empty."
        )

    encrypted_data = arrays[0].tobytes()

    return decrypt_and_deserialize_model(
        encrypted_data,
        key,
    )