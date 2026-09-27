"""
AES-256 encryption for federated learning model updates.
"""

import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


AES_KEY_SIZE = 32   # 32 bytes = 256 bits
NONCE_SIZE = 12     # 96-bit nonce recommended for AES-GCM


def generate_key() -> bytes:
    """Generate a random 256-bit AES key."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_model_update(data: bytes, key: bytes) -> bytes:
    """
    Encrypt serialized model-update bytes using AES-256-GCM.

    The returned format is:

        nonce + ciphertext + authentication tag
    """

    if not isinstance(data, bytes):
        raise TypeError("Model update must be bytes.")

    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must be exactly 32 bytes."
        )

    nonce = os.urandom(NONCE_SIZE)

    aes = AESGCM(key)

    encrypted_data = aes.encrypt(
        nonce,
        data,
        None
    )

    return nonce + encrypted_data