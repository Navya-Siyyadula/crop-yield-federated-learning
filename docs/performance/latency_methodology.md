# Latency methodology

All durations use the monotonic `time.perf_counter()` clock and are recorded in milliseconds. `Timer` supports start/stop and context-manager measurement; `LatencyCollector` accumulates repeated named stages.

## Stage boundaries

- **Local training (`training_latency_ms`)**: should cover the client's local model fit call from immediately before training starts through its return. The inspected repository has no implemented Flower client or local-training function, so this field remains blank; no client training time is claimed.
- **Serialization (`serialization_latency_ms`)**: `serialize_model_update(weights)` entry to NPZ bytes returned in `serialize_and_encrypt_weights`. Deserialization has a separate `deserialization_latency_ms` field.
- **Encryption (`encryption_latency_ms`)**: the AES-256-GCM `encrypt_model_update` call only. In the current server-side strategy, the measured encryption is the new global model packaged for the next round. No Flower client exists here to report client upload encryption time.
- **Decryption (`decryption_latency_ms`)**: the AES-GCM `decrypt_model_update` call only for each received client payload; repeated values are summed for the round. NPZ unpacking is reported separately as deserialization.
- **Aggregation (`aggregation_latency_ms`)**: the existing sample-count-weighted `fedavg(client_updates)` call, from invocation to returned aggregated arrays.
- **Communication (`communication_latency_ms`)**: intentionally blank. Flower server timing cannot isolate network transport from client scheduling and work in the current implementation. The implementation does not relabel combined interaction time as pure communication.
- **Round (`total_round_latency_ms`)**: starts when the Flower strategy's `configure_fit` begins and ends when `aggregate_fit` has received and aggregated the client fit responses and prepared encrypted global parameters. It includes client work, Flower coordination, transport, server processing, and response preparation; it is a full round interaction wall time, not network-only time. It is recorded only when those strategy hooks execute.
- **Security overhead (`security_overhead_ms`)**: sum of the measured AES encryption and decryption stages when available; serialization costs remain separately visible.

The server strategy writes one row per completed round to `results/performance_metrics.csv`. It logs absent client-only measurements as blank values. It does not perform duplicate training, encryption, or aggregation to obtain timings. The separate `measure_overhead.py` benchmark serializes weights from the repository's trained Keras LSTM artifact and checks decryption round-trip correctness; it measures crypto operations independently of a federated round.

## ELEI normalization

ELEI uses the formula and validation described in [energy methodology](energy_methodology.md). Energy and latency references must be explicitly supplied and positive; no built-in baseline or accuracy value is used. The logger records both references and an optional baseline/source label. ELEI is omitted when energy or references are unavailable.
