"""
Model-update serialization for the Crop Yield Federated Learning project.

Contract
--------
Input:
    A Keras/TensorFlow model update represented by model.get_weights(),
    i.e. a list/tuple of NumPy arrays.

Output:
    bytes suitable for the AES-256 encryption module.

Format:
    NumPy NPZ archive created with np.savez_compressed().
    Each weight is stored as array_0, array_1, ...
    and format_version stores the serialization version.

Why NPZ:
    - preserves array shape and dtype
    - deterministic ordering by array index
    - does not require pickle
    - uses only NumPy for serialization/deserialization
    - produces bytes directly for AES encryption

Important:
    sample_count is NOT included in these bytes. It should be sent/stored
    separately as FL metadata because FedAvg needs the client sample count.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np


FORMAT_VERSION = 1


def serialize_model_update(weights: Sequence[np.ndarray]) -> bytes:
    """
    Convert a Keras model.get_weights() list into bytes.

    Parameters
    ----------
    weights:
        List/tuple of NumPy arrays returned by model.get_weights().

    Returns
    -------
    bytes:
        Serialized model-update bytes for the AES-256 layer.
    """
    if not isinstance(weights, (list, tuple)):
        raise TypeError("weights must be a list or tuple of NumPy arrays")

    buffer = io.BytesIO()

    payload = {
        f"array_{i}": np.asarray(weight)
        for i, weight in enumerate(weights)
    }
    payload["format_version"] = np.array([FORMAT_VERSION], dtype=np.int32)

    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def deserialize_model_update(data: bytes) -> list[np.ndarray]:
    """
    Convert serialized model-update bytes back into a Keras-compatible
    list of NumPy arrays.

    Parameters
    ----------
    data:
        Bytes produced by serialize_model_update().

    Returns
    -------
    list[np.ndarray]:
        Weight arrays in their original order.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    raw = bytes(data)

    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if "format_version" not in archive.files:
                raise ValueError("Missing serialization format version")

            version = int(np.asarray(archive["format_version"]).reshape(-1)[0])
            if version != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported serialization format version: {version}"
                )

            names = [
                name for name in archive.files
                if name.startswith("array_")
            ]

            def index(name: str) -> int:
                return int(name.split("_", 1)[1])

            names.sort(key=index)

            if not names:
                raise ValueError("Serialized update contains no weight arrays")

            return [np.array(archive[name], copy=True) for name in names]

    except (OSError, ValueError, KeyError, IndexError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Unsupported"):
            raise
        raise ValueError("Invalid serialized model-update bytes") from exc


def round_trip_test(weights: Sequence[np.ndarray]) -> bool:
    """
    Verify serialize -> deserialize preserves values, shapes and dtypes.
    """
    serialized = serialize_model_update(weights)
    recovered = deserialize_model_update(serialized)

    if len(weights) != len(recovered):
        return False

    return all(
        np.array_equal(np.asarray(original), restored)
        and np.asarray(original).dtype == restored.dtype
        for original, restored in zip(weights, recovered)
    )


# Optional aliases for a simple team-wide interface.
serialize = serialize_model_update
deserialize = deserialize_model_update


if __name__ == "__main__":
    # Smoke test using the exact shapes of the current LSTM:
    # Input(1,38) -> LSTM(64) -> Dense(32) -> Dense(1)
    shapes = [
        (38, 256),  # LSTM kernel
        (64, 256),  # LSTM recurrent kernel
        (256,),      # LSTM bias
        (64, 32),    # Dense(32) kernel
        (32,),       # Dense(32) bias
        (32, 1),     # Dense(1) kernel
        (1,),        # Dense(1) bias
    ]

    sample_weights = [
        np.zeros(shape, dtype=np.float32)
        for shape in shapes
    ]

    serialized = serialize_model_update(sample_weights)
    recovered = deserialize_model_update(serialized)

    assert round_trip_test(sample_weights)
    print("Serialization test: PASS")
    print(f"Number of weight arrays: {len(recovered)}")
    print(f"Serialized size: {len(serialized)} bytes")
    print(f"SHA-256: {__import__('hashlib').sha256(serialized).hexdigest()}")