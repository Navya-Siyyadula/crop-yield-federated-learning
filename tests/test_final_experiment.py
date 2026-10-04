from src.evaluation.final_experiment import (
    CLIENT_SIZES,
    crowding_distance,
    dominates,
    load_frozen_data,
    nondominated_fronts,
)


def test_frozen_dataset_and_four_client_sizes():
    data = load_frozen_data()
    assert [len(data[name][0]) for name in ("train", "val", "test")] == [350, 75, 75]
    assert CLIENT_SIZES == (88, 88, 88, 86)
    assert sum(CLIENT_SIZES) == len(data["train"][0])


def test_nsga_ii_minimization_dominance_and_fronts():
    values = [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0), (3.0, 5.0)]
    assert dominates((1, 2), (2, 3))
    assert not dominates((1, 2), (1, 2))
    fronts = nondominated_fronts(values)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]
    distances = crowding_distance(fronts[0], values)
    assert sum(value == float("inf") for value in distances.values()) >= 2


def test_discrete_nsga_decision_variable_domain():
    epochs = range(1, 6)
    batches = (16, 32, 64)
    candidates = [(epoch, batch) for epoch in epochs for batch in batches]
    assert len(candidates) == 15
    assert all(1 <= epoch <= 5 and batch in batches for epoch, batch in candidates)
