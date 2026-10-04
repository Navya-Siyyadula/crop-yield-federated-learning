# Latency methodology

All durations use the monotonic `time.perf_counter()` clock and are recorded in milliseconds. `Timer` supports start/stop and context-manager measurement; `LatencyCollector` accumulates repeated named stages.

## Stage boundaries

For the final comparison runner, latency is the sum of the 20 actual Flower round durations (`perf_counter`) for each four-client candidate. The server's `configure_fit` to `aggregate_fit` boundary includes client fitting, Flower coordination and transport over loopback, serialization, AES encryption/decryption, and FedAvg aggregation. It is local end-to-end round wall time, not communication-only latency or WAN latency. The edge–cloud comparator is one Flower round. Centralized latency times the complete centralized fit task. These scopes must not be conflated.

- **Local training (`training_latency_ms`)**: each Flower client measures the local model fit call from immediately before training through its return. Per-client data is recorded in `results/client_performance.csv`.
- **Serialization (`serialization_latency_ms`)**: `serialize_model_update(weights)` entry to NPZ bytes returned in `serialize_and_encrypt_weights`. Deserialization has a separate `deserialization_latency_ms` field.
- **Encryption (`encryption_latency_ms`)**: the AES-256-GCM encryption call for local updates and the aggregated global model. Client upload encryption is included in per-client telemetry.
- **Decryption (`decryption_latency_ms`)**: the AES-GCM decryption call for received client payloads; repeated values are summed for the round. NPZ unpacking is reported separately as deserialization.
- **Aggregation (`aggregation_latency_ms`)**: the existing sample-count-weighted `fedavg(client_updates)` call, from invocation to returned aggregated arrays.
- **Communication (`communication_latency_ms`)**: intentionally blank. Flower server timing cannot isolate network transport from client scheduling and work in the current implementation. The implementation does not relabel combined interaction time as pure communication.
- **Round (`total_round_latency_ms`)**: starts when the Flower strategy's `configure_fit` begins and ends when `aggregate_fit` has received and aggregated the client fit responses and prepared encrypted global parameters. It includes client work, Flower coordination, transport, server processing, and response preparation; it is a full round interaction wall time, not network-only time. It is recorded only when those strategy hooks execute.
- **Security overhead (`security_overhead_ms`)**: sum of the measured AES encryption and decryption stages when available; serialization costs remain separately visible.

The final experiment writes round-level measurements to `results/fl_round_metrics.csv` with candidate identifiers, and client-level timing/payload/scaler values to `results/client_performance.csv`. It also appends the strategy's lower-level timing log to `results/final_fl_normalized_performance_metrics.csv`. The experiment does not repeat training, encryption, or aggregation to obtain timing values. The separate `measure_overhead.py` benchmark measures crypto operations independently of a federated round.

## ELEI normalization

ELEI uses the formula and validation described in [energy methodology](energy_methodology.md). Energy and latency references must be explicitly supplied and positive; no built-in baseline or accuracy value is used. The logger records both references and an optional baseline/source label. ELEI is omitted when energy or references are unavailable.
