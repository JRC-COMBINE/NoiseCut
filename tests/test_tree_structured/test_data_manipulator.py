import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noisecut.tree_structured.data_manipulator import DataManipulator

THIS_DIR = Path(__file__).parent


@pytest.fixture
def data():
    df = pd.read_csv(
        str(THIS_DIR / "../test_model/toy_example/toy_example_complete"),
        header=None,
        delimiter="    ",
        skiprows=1,
        engine="python",
    )

    X = df.iloc[:, :-1].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    return X, y


@pytest.mark.parametrize(
    "train_size, train_test_shape",
    [
        (5, [(3, 6), (3,), (61, 6), (61,)]),
        (20, [(12, 6), (12,), (52, 6), (52,)]),
        (33, [(21, 6), (21,), (43, 6), (43,)]),
    ],
)
def test_split_data(data, train_size, train_test_shape):
    x, y = data

    cls = DataManipulator()
    x_train, y_train, x_test, y_test = cls.split_data(
        x, y, percentage_training_data=train_size
    )

    assert np.shape(x_train) == train_test_shape[0]
    assert np.shape(y_train) == train_test_shape[1]
    assert np.shape(x_test) == train_test_shape[2]
    assert np.shape(y_test) == train_test_shape[3]


@pytest.mark.parametrize("percent_train", [10, 35, 40])
def test_split_data_unique(data, percent_train):
    x, y = data

    cls = DataManipulator()
    x_train, y_train, x_test, y_test = cls.split_data(
        x, y, percentage_training_data=percent_train
    )

    n_train = len(y_train)
    n_test = len(y_test)
    for id_data in range(len(y)):
        count = 0
        for i in range(n_train):
            if np.array_equal(x[id_data], x_train[i]):
                count += 1
                assert y[id_data] == y_train[i]

        for i in range(n_test):
            if np.array_equal(x[id_data], x_test[i]):
                count += 1
                assert y[id_data] == y_test[i]

        assert count == 1


@pytest.mark.parametrize("percent_noise", [3, 5, 10])
def test_get_noisy_data(data, percent_noise):
    x, y = data

    cls = DataManipulator()
    _, y_noisy = cls.get_noisy_data(x, y, percent_noise)

    n_data = len(y)
    n_to_change = math.ceil(n_data * percent_noise / 100)

    n_noisey_data = 0
    for i in range(n_data):
        if y[i] != y_noisy[i]:
            n_noisey_data += 1

    assert n_noisey_data == n_to_change
