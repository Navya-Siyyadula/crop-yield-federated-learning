# Energy methodology

## Meaning and status

Energy is the energy consumed by the hardware for an FL operation, expressed in joules. Every record has an `energy_status` of `measured`, `estimated`, or `unavailable`, plus an `energy_method` describing its source. Predictive quality metrics (MAE, RMSE, and R²) are independent of energy and are not part of ELEI.

The final comparison uses CodeCarbon task tracking around centralized training and each complete Flower round. It records whole-machine software estimates for CPU/GPU/RAM; these are labeled `estimated`, not electrical meter readings. The generic default `EnergyMeter` remains unavailable unless a provider is configured; elapsed time is never converted to joules.

For each candidate, the runner sums 20 per-round `energy_consumed` values (kWh converted to joules), retaining `energy_status=estimated`. The centralized and one-round logical edge/cloud rows use their own CodeCarbon task estimates. Emissions (kg CO2e) are never treated as energy. If a candidate lacks any round energy estimate, its energy objective is unavailable and no Pareto front/EMO selection is claimed. No fallback value is imputed.

## Measurement limitations on Windows

CodeCarbon estimates are software-derived and host scoped; they are not calibrated wall-power measurements. The local Flower run shares one Windows host and includes background host activity within the tracker scope. Estimates can vary with host load and should not be presented as direct electrical measurements. Unit tests use a fake provider only to verify status handling and do not read hardware.

## ELEI and reference normalization

For energy `E` in joules and latency `L` in milliseconds:

```text
ELEI = w_energy * (E / E_ref) + w_latency * (L / L_ref)
```

Weights default to 0.5 each and must be non-negative and sum to 1. Both positive references (`E_ref`, `L_ref`) are mandatory. No reference is silently selected. If a baseline run supplies references, callers must explicitly pass those numeric values and identify `reference_source` as that baseline; the values and source are logged on each row. ELEI is left blank when energy or either reference is unavailable. A score below 1 means lower normalized combined resource use than the reference; 1 means equal; above 1 means higher. Estimated energy can be used only when explicitly supplied and remains marked estimated.
