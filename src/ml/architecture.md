# LSTM Model Architecture

## Purpose

The LSTM model is used for crop-yield prediction as a regression task. The target variable is `yield_kg_per_hectare`.

## Input

- Number of processed features: 38
- Input shape: `(1, 38)`
- One sample is represented as one timestep containing 38 features.

## Architecture

```text
Input
(1 timestep, 38 features)
        ↓
LSTM(64)
        ↓
Dense(32, ReLU)
        ↓
Dense(1)
        ↓
Predicted Crop Yield
## Interface Contract

- Input shape: `(batch_size, 1, 38)`
- Output shape: `(batch_size, 1)`
- Input: 38 processed features represented as one timestep
- Output: continuous `yield_kg_per_hectare`
- Task: Regression

The dataset does not contain genuine multi-timestep sequences. Therefore, this implementation is treated as a single-timestep LSTM experiment and does not claim to learn temporal dependencies.
