# EMO and NSGA-II methodology

## Purpose and objectives

Evolutionary multi-objective optimization (EMO) searches for configurations that trade off multiple objectives. This study uses NSGA-II to minimize whole-run four-client FL latency and energy. The optimizer does not combine them into a weighted scalar. MAE, RMSE, and R² are separately evaluated on validation data for candidate reporting and are not a third objective.

## Decision variables

The discrete genotype is `(local_epochs, batch_size)`, where local epochs are integers 1–5 and batch size is one of 16, 32, or 64. Four fixed clients partition the frozen 350 training samples as 88/88/88/86. The optimizer varies only controls actually passed to local model training. Rounds, model architecture, dataset, seed, no-shuffle training order, and client participation remain fixed within a run.

## Candidate measurement and objective values

One `StandardScaler` is fitted once from the 350 training targets across all clients. Its parameters are sent to every client; no client independently fits a target scaler. Client training targets and model outputs share the same normalized space. The server inverse-transforms validation predictions, and final selected/baseline model predictions, with that scaler before computing MAE/RMSE/R² in kg/ha. Validation/test values never fit the scaler. Test data is opened only after selection.

Every candidate is executed as four-client sample-count-weighted Flower FedAvg for 20 rounds, with serialized model updates encrypted/decrypted using AES-256-GCM. Client local epoch count remains fixed within each candidate and there is no client-local validation checkpointing. Latency is the sum of measured full Flower round durations; it includes local work, coordination, cryptographic processing, and aggregation. It is not WAN/network latency. CodeCarbon task energy is used only when available, converted from kWh to joules and labeled estimated. CodeCarbon emissions (kg CO2e) are never converted or mislabeled as joules. If energy is unavailable, the Pareto front and EMO selection are withheld.

## NSGA-II and Pareto dominance

For minimization, candidate A dominates B if A is no worse on both latency and energy and is strictly better on at least one. NSGA-II sorts measured candidates into non-dominated fronts, uses crowding distance to preserve objective-space spread, selects parents by rank and crowding, applies discrete crossover/mutation, and uses elitist environmental selection. No prototype constants, fabricated measurements, weighted objective, or surrogate values are used. The CSV records candidate, generation, both measured objectives, prediction metrics, Pareto status, and selection status.

## Selection policy

Select the Pareto point with the smallest Euclidean distance to the ideal point after independently min–max normalizing latency and energy across the observed Pareto front. This knee policy gives both objectives equal scale without combining them into a weighted objective. Candidate prediction quality is reported on validation data and does not drive Pareto ranking. If energy is unavailable, no Pareto selection is made; the final FL comparison reports the predeclared `(local_epochs=1, batch_size=32)` baseline instead. The completed normalized-target run measured 13 unique candidates under population 6/generations 3; genetic crossover/mutation generated repeated genotypes before all 15 possible combinations were measured.

## Prediction protocol

All final prediction metrics use the same frozen `X_test.csv`/`y_test.csv`, target `yield_kg_per_hectare`, saved preprocessing, and MAE/RMSE/R² definitions. Candidate prediction-quality columns and round curves use the frozen validation set. The selected (or fixed fallback) global model is evaluated on the test set only after configuration selection; test metrics do not influence training, Pareto ranking, or selection.

## Limitations and adaptations

The LSTM input is `(samples, 1, 38)`, a single-timestep LSTM-based regressor without learned multi-step temporal dependencies. The observed Pareto front contains one candidate, so its normalized objective values are both zero at the observed ideal point. Edge–cloud execution and four-client Flower are one-host logical simulations; no physical edge server or WAN latency is measured. Energy is software-estimated with CodeCarbon, not electrical telemetry. No arbitrary objective constants or weighted latency-energy sum are used.
