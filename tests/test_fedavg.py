import numpy as np

from federated.fedavg import fedavg
from src.federated.rounds import run_federated_training



def test_fedavg_equal_clients():

    client_updates = [
        ([np.array([1.0, 2.0])], 100),
        ([np.array([3.0, 4.0])], 100),
    ]

    result = fedavg(client_updates)

    expected = np.array([2.0, 3.0])

    np.testing.assert_allclose(
        result[0],
        expected
    )


def test_fedavg_sample_weighting():

    client_updates = [
        ([np.array([1.0, 1.0])], 100),
        ([np.array([3.0, 3.0])], 300),
    ]

    result = fedavg(client_updates)

    # (1*100 + 3*300) / 400 = 2.5

    expected = np.array([2.5, 2.5])

    np.testing.assert_allclose(
        result[0],
        expected
    )


def test_fedavg_multiple_layers():

    client_updates = [
        (
            [
                np.array([[1.0, 2.0]]),
                np.array([3.0, 4.0]),
            ],
            100,
        ),
        (
            [
                np.array([[3.0, 4.0]]),
                np.array([5.0, 6.0]),
            ],
            100,
        ),
    ]

    result = fedavg(client_updates)

    np.testing.assert_allclose(
        result[0],
        np.array([[2.0, 3.0]])
    )

    np.testing.assert_allclose(
        result[1],
        np.array([4.0, 5.0])
    )


def test_two_federated_rounds():

    def client_1(global_weights):

        updated_weights = [
            global_weights[0] + 1
        ]

        return updated_weights, 100


    def client_2(global_weights):

        updated_weights = [
            global_weights[0] + 3
        ]

        return updated_weights, 100


    initial_weights = [
        np.array([0.0, 0.0])
    ]


    final_weights, history = run_federated_training(
        initial_weights,
        [
            client_1,
            client_2
        ],
        num_rounds=2
    )


    assert len(history) == 2

    np.testing.assert_allclose(
        final_weights[0],
        np.array([4.0, 4.0])
    )
