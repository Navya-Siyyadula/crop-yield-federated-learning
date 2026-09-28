"""
AES-256 key management for federated learning.

The encryption key is supplied through the FL_AES_KEY
environment variable and is never stored in source code.
"""

import os
import base64

from src.security.encryption import (
    AES_KEY_SIZE,
    generate_key,
)


ENV_KEY_NAME = "FL_AES_KEY"


def generate_session_key() -> bytes:
    """Generate a fresh AES-256 session key."""
    return generate_key()


def encode_key(key: bytes) -> str:
    """Encode an AES key for an environment variable."""
    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must be exactly 32 bytes."
        )

    return base64.b64encode(key).decode("ascii")


def decode_key(encoded_key: str) -> bytes:
    """Decode an environment-variable AES key."""
    try:
        key = base64.b64decode(
            encoded_key,
            validate=True,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid AES key encoding."
        ) from exc

    if len(key) != AES_KEY_SIZE:
        raise ValueError(
            "AES-256 key must decode to exactly 32 bytes."
        )

    return key


def load_key_from_environment() -> bytes:
    """Load the shared FL AES key from the environment."""

    encoded_key = os.environ.get(ENV_KEY_NAME)

    if not encoded_key:
        raise RuntimeError(
            f"{ENV_KEY_NAME} environment variable is not set."
        )

    return decode_key(encoded_key)