"""
Flower's Parameters object is built around lists of numpy ndarrays
(designed for neural net weight tensors). A RandomForestRegressor has no
weight tensors, so we serialize the whole fitted model to bytes (pickle)
and wrap those bytes as a single uint8 ndarray - that's the standard
workaround used for non-parametric models in Flower.

Both client.py and server_strategy.py import this module.
"""

import pickle
import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays


def model_to_parameters(model) -> Parameters:
    raw = pickle.dumps(model)
    arr = np.frombuffer(raw, dtype=np.uint8).copy()
    return ndarrays_to_parameters([arr])


def parameters_to_model(parameters: Parameters):
    ndarrays = parameters_to_ndarrays(parameters)
    raw = ndarrays[0].tobytes()
    return pickle.loads(raw)


def empty_parameters() -> Parameters:
    return ndarrays_to_parameters([np.zeros(1, dtype=np.uint8)])
