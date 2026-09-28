import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.security.encrypted_update import (
    create_encryption_key,
    serialize_model,
    serialize_and_encrypt_model,
    decrypt_and_deserialize_model,
)


def main():

    # Synthetic model used only to measure
    # encryption implementation overhead.
    rng = np.random.RandomState(42)

    X = rng.rand(100, 10)
    y = rng.rand(100)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X, y)

    key = create_encryption_key()

    # Measure plaintext serialization
    start = time.perf_counter()

    serialized_model = serialize_model(model)

    serialization_time = (
        time.perf_counter() - start
    )

    # Measure encryption
    start = time.perf_counter()

    encrypted_model = serialize_and_encrypt_model(
        model,
        key,
    )

    encryption_time = (
        time.perf_counter() - start
    )

    # Measure decryption
    start = time.perf_counter()

    restored_model = decrypt_and_deserialize_model(
        encrypted_model,
        key,
    )

    decryption_time = (
        time.perf_counter() - start
    )

    plaintext_size = len(
        serialized_model
    )

    encrypted_size = len(
        encrypted_model
    )

    size_overhead = (
        (
            encrypted_size
            - plaintext_size
        )
        / plaintext_size
    ) * 100

    print(
        "Encryption overhead measurement"
    )
    print(
        "--------------------------------"
    )
    print(
        f"Plain serialized size: "
        f"{plaintext_size} bytes"
    )
    print(
        f"Encrypted size: "
        f"{encrypted_size} bytes"
    )
    print(
        f"Size overhead: "
        f"{size_overhead:.2f}%"
    )
    print(
        f"Serialization time: "
        f"{serialization_time:.6f} seconds"
    )
    print(
        f"Encryption time: "
        f"{encryption_time:.6f} seconds"
    )
    print(
        f"Decryption time: "
        f"{decryption_time:.6f} seconds"
    )
    print(
        f"Restored trees: "
        f"{len(restored_model.estimators_)}"
    )


if __name__ == "__main__":
    main()