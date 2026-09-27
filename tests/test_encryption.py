import pytest

from security.encryption import (
    generate_key,
    encrypt_model_update,
)

from security.decryption import (
    decrypt_model_update,
)

def test_key_is_256_bits():
    key = generate_key()

    assert len(key) == 32


def test_encrypt_decrypt():
    key = generate_key()
    original = b"test model update"

    encrypted = encrypt_model_update(
        original,
        key
    )

    decrypted = decrypt_model_update(
        encrypted,
        key
    )

    assert decrypted == original


def test_encryption_changes_data():
    key = generate_key()
    original = b"test model update"

    encrypted = encrypt_model_update(
        original,
        key
    )

    assert encrypted != original


def test_wrong_key_fails():
    key = generate_key()
    wrong_key = generate_key()

    original = b"test model update"

    encrypted = encrypt_model_update(
        original,
        key
    )

    with pytest.raises(Exception):
        decrypt_model_update(
            encrypted,
            wrong_key
        )


def test_corrupted_data_fails():
    key = generate_key()
    original = b"test model update"

    encrypted = encrypt_model_update(
        original,
        key
    )

    corrupted = bytearray(encrypted)
    corrupted[-1] ^= 1

    with pytest.raises(Exception):
        decrypt_model_update(
            bytes(corrupted),
            key
        )