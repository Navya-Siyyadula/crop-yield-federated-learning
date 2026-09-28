"""
AES-256 decryption for federated learning model updates.
"""

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.security.encryption import (
    AES_KEY_SIZE,
    NONCE_SIZE,
)

def decrypt_model_update(
    encrypted_data: bytes,
    key: bytes
) -> bytes:
    """
    Decrypt AES-256-GCM encrypted model-update bytes.
    """

    if not isinstance(encrypted_data, bytes):
        raise TypeError(
            "Encrypted data must be bytes."
        )

    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must be exactly 32 bytes."
        )

    if len(encrypted_data) <= NONCE_SIZE:
        raise ValueError(
            "Encrypted data is too short."
        )

    nonce = encrypted_data[:NONCE_SIZE]
    ciphertext = encrypted_data[NONCE_SIZE:]

    aes = AESGCM(key)

    return aes.decrypt(
        nonce,
        ciphertext,
        None
    )