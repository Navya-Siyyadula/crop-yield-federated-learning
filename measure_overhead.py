import csv
import time
from pathlib import Path

from src.federated.serialize import serialize_model_update
from src.security.encryption import generate_key, encrypt_model_update
from src.security.decryption import decrypt_model_update
from tensorflow import keras


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "src" / "ml" / "lstm_crop_yield_model.keras"
RESULT_FILE = PROJECT_ROOT / "results" / "encryption_overhead.csv"

WARMUP_RUNS = 5
MEASUREMENT_RUNS = 30


# -----------------------------
# Load model update
# -----------------------------

def load_model_update():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained LSTM model not found: {MODEL_FILE}"
        )
    model = keras.models.load_model(MODEL_FILE)
    return serialize_model_update(model.get_weights())


# -----------------------------
# Measure encryption
# -----------------------------

def measure_encryption(data, key):
    start = time.perf_counter()

    encrypted_data = encrypt_model_update(data, key)

    end = time.perf_counter()

    encryption_time_ms = (end - start) * 1000

    return encrypted_data, encryption_time_ms


# -----------------------------
# Measure decryption
# -----------------------------

def measure_decryption(encrypted_data, key):
    start = time.perf_counter()

    decrypted_data = decrypt_model_update(
        encrypted_data,
        key
    )

    end = time.perf_counter()

    decryption_time_ms = (end - start) * 1000

    return decrypted_data, decryption_time_ms


# -----------------------------
# Main benchmark
# -----------------------------

def main():

    print("Starting Encryption Overhead Measurement...")

    # Load serialized model update
    serialized_data = load_model_update()

    print(
        f"Serialized update size: "
        f"{len(serialized_data)} bytes"
    )

    # Generate AES-256 key
    key = generate_key()

    # Warm-up runs
    print(f"Running {WARMUP_RUNS} warm-up runs...")

    for _ in range(WARMUP_RUNS):

        encrypted_data, _ = measure_encryption(
            serialized_data,
            key
        )

        decrypted_data, _ = measure_decryption(
            encrypted_data,
            key
        )

        if decrypted_data != serialized_data:
            raise ValueError(
                "Decryption verification failed."
            )

    # Measurement
    print(
        f"Running {MEASUREMENT_RUNS} measurement runs..."
    )

    results = []

    for run_number in range(
        1,
        MEASUREMENT_RUNS + 1
    ):

        encrypted_data, encryption_time = (
            measure_encryption(
                serialized_data,
                key
            )
        )

        decrypted_data, decryption_time = (
            measure_decryption(
                encrypted_data,
                key
            )
        )

        # Verify correctness
        if decrypted_data != serialized_data:
            raise ValueError(
                f"Verification failed at run {run_number}"
            )

        total_overhead = (
            encryption_time +
            decryption_time
        )

        results.append({
            "run": run_number,
            "serialized_size_bytes": len(serialized_data),
            "encrypted_size_bytes": len(encrypted_data),
            "encryption_time_ms": encryption_time,
            "decryption_time_ms": decryption_time,
            "security_overhead_ms": total_overhead
        })

    # Create results directory
    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Write CSV
    file_exists = RESULT_FILE.exists() and RESULT_FILE.stat().st_size > 0
    fieldnames = [
        "run",
        "serialized_size_bytes",
        "encrypted_size_bytes",
        "encryption_time_ms",
        "decryption_time_ms",
        "security_overhead_ms",
    ]
    if file_exists:
        with open(RESULT_FILE, newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header != fieldnames:
            raise ValueError(f"Unexpected columns in {RESULT_FILE}")

    with open(
        RESULT_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    # Calculate averages
    avg_encryption = sum(
        r["encryption_time_ms"]
        for r in results
    ) / len(results)

    avg_decryption = sum(
        r["decryption_time_ms"]
        for r in results
    ) / len(results)

    avg_overhead = sum(
        r["security_overhead_ms"]
        for r in results
    ) / len(results)

    print("\n========== RESULTS ==========")

    print(
        f"Average Encryption Time : "
        f"{avg_encryption:.6f} ms"
    )

    print(
        f"Average Decryption Time : "
        f"{avg_decryption:.6f} ms"
    )

    print(
        f"Average Security Overhead : "
        f"{avg_overhead:.6f} ms"
    )

    print(
        f"\nResults saved to: "
        f"{RESULT_FILE}"
    )


if __name__ == "__main__":
    main()
