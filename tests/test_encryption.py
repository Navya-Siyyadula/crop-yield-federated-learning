"""
Unit tests for AES-256-GCM encryption and decryption.
"""

import pytest

from src.security.encryption import generate_key, encrypt_model_update, AES_KEY_SIZE, NONCE_SIZE
from src.security.decryption import decrypt_model_update


def test_generate_key():
    key = generate_key()
    assert isinstance(key, bytes)
    assert len(key) == AES_KEY_SIZE


def test_encryption_decryption_round_trip():
    key = generate_key()
    original_data = b"Simulated LSTM serialized weights test payload 1234567890"

    encrypted_data = encrypt_model_update(original_data, key)
    assert isinstance(encrypted_data, bytes)
    assert len(encrypted_data) > len(original_data) + NONCE_SIZE

    decrypted_data = decrypt_model_update(encrypted_data, key)
    assert decrypted_data == original_data


def test_decryption_tamper_detection():
    key = generate_key()
    original_data = b"Test payload for integrity check"
    encrypted = bytearray(encrypt_model_update(original_data, key))

    # Tamper with the ciphertext
    encrypted[-1] ^= 0x01

    with pytest.raises(Exception):
        decrypt_model_update(bytes(encrypted), key)
