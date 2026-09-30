# Energy methodology

## Meaning and status

Energy is the energy consumed by the hardware for an FL operation, expressed in joules. Every record has an `energy_status` of `measured`, `estimated`, or `unavailable`, plus an `energy_method` describing its source. Predictive quality metrics (MAE, RMSE, and R²) are independent of energy and are not part of ELEI.

The current repository has no supported power/energy telemetry provider or calibrated CPU power model. The default `EnergyMeter` therefore writes `energy_status=unavailable`, a blank `energy_joules`, and an explanatory method/detail. It does not infer energy from elapsed time. `EnergyReading` accepts measured or estimated values only when a caller explicitly supplies both the value and method. There is no default estimator and no fabricated joule fallback.

## Measurement limitations on Windows

The project's Windows environment does not expose a portable, repository-configured electrical energy source. Windows processor energy counters, when present, are hardware and driver dependent; ordinary process CPU utilization and elapsed time do not establish joules. No telemetry package or device-specific interface is configured here. Consequently energy for this implementation is unavailable until a validated meter/provider is supplied. Unit tests use a fake provider only to verify status handling and do not read hardware.

## ELEI and reference normalization

For energy `E` in joules and latency `L` in milliseconds:

```text
ELEI = w_energy * (E / E_ref) + w_latency * (L / L_ref)
```

Weights default to 0.5 each and must be non-negative and sum to 1. Both positive references (`E_ref`, `L_ref`) are mandatory. No reference is silently selected. If a baseline run supplies references, callers must explicitly pass those numeric values and identify `reference_source` as that baseline; the values and source are logged on each row. ELEI is left blank when energy or either reference is unavailable. A score below 1 means lower normalized combined resource use than the reference; 1 means equal; above 1 means higher. Estimated energy can be used only when explicitly supplied and remains marked estimated.
