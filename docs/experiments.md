# Final experiment protocol

Install the tested runtime with `python -m pip install -r requirements-experiment.txt`, then run:

```powershell
python scripts/run_final_comparison.py --rounds 20 --central-epochs 20 --population 6 --generations 3
```

The runner checks the frozen 350/75/75 split, target name, 38-feature input, and four frozen training partitions (88/88/88/86). It fits one `StandardScaler` from all 350 training targets before any client training. That same training-fitted mean/scale is passed to every client. Validation targets are transformed with the shared scaler but never used to fit it. The held-out test CSVs are read only after candidate selection.

The centralized LSTM uses the same scaler and restores the minimum-validation-loss checkpoint from at most 20 epochs. The logical edge/cloud comparison uses one Flower round. The federated baseline uses 4 clients and 20 Flower FedAvg rounds at 1 local epoch and batch size 32. The NSGA-II search retains the established local-epoch range (1–5), batch sizes (16/32/64), population (6), and generations (3). Clients use the fixed local epoch count for every round; no local validation checkpointing is applied. All prediction metrics are calculated after inverse transformation in kg/ha.

Flower runs on one host over loopback. Latency sums measured round durations, including client fitting, coordination, serialization, encryption/decryption, and aggregation. CodeCarbon supplies estimated whole-machine energy per round; missing energy is not replaced with a fabricated value, and no Pareto result is claimed when a candidate objective is unavailable. Artifacts include candidate, round, client, comparison, configuration, and plot outputs in `results/`. See `docs/emo_methodology.md` for selection details.
