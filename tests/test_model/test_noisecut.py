from pathlib import Path

import numpy
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from noisecut.max_cut.solvers import MaxCutSolvers
from noisecut.model.noisecut_coder import (
    CoderNoiseCut,
    Metric,
)
from noisecut.model.noisecut_model import NoiseCut

THIS_DIR = str(Path(__file__).parent)

X_train = [
    [1, 0, 1, 1, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
]
y_train = [1, 0, 1, 0]


@pytest.fixture
def training_data():
    dim = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_training",
        header=None,
        delimiter="    ",
        nrows=1,
        engine="python",
    )
    df = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_training",
        header=None,
        delimiter="    ",
        skiprows=1,
        engine="python",
    )

    n_each_box = dim.iloc[0, :].values.tolist()
    X = df.iloc[:, :-1].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    return n_each_box, X, y


@pytest.fixture
def test_data():
    df = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_test",
        header=None,
        delimiter="    ",
        engine="python",
    )

    X = df.iloc[:, :-1].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    return X, y


@pytest.fixture
def true_data():
    dim = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_complete",
        header=None,
        delimiter="    ",
        nrows=1,
        engine="python",
    )
    df = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_complete",
        header=None,
        delimiter="    ",
        skiprows=1,
        engine="python",
    )

    n_each_box = dim.iloc[0, :].values.tolist()
    X = df.iloc[:, :-1].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    return n_each_box, X, y


@pytest.fixture
def more_data():
    df = pd.read_csv(
        THIS_DIR + "/toy_example/toy_example_more_training",
        header=None,
        delimiter="    ",
        engine="python",
    )

    X = df.iloc[:, :-1].values.tolist()
    y = df.iloc[:, -1].values.tolist()

    return X, y


@pytest.fixture
def noisecut_model(training_data):
    n_each_box, X, y = training_data

    mdl = CoderNoiseCut(n_each_box)
    mdl.coder_set_training_data(X, y)

    return mdl


@pytest.fixture
def fitted_noisecut_model(training_data):
    n_each_box, X, y = training_data

    mdl = NoiseCut(n_each_box)
    mdl.fit(X, y)

    return mdl


@pytest.mark.parametrize(
    "x, y, index, n",
    [
        ([X_train[0]], 1, 189, [0, 1]),
        (X_train[0], [[1]], 189, [0, 1]),
        (X_train[1], 0, 102, [1, 0]),
    ],
)
def test_set_training_data_one_data(x, y, index, n):
    mdl = CoderNoiseCut([3, 3, 2])
    mdl.coder_set_training_data(x, y)

    assert mdl.y_number_of_0[index] == n[0]
    assert mdl.y_number_of_1[index] == n[1]


def test_set_training_data():
    mdl = CoderNoiseCut([3, 3, 2])

    mdl.coder_set_training_data(X_train, y_train)

    assert mdl.y_number_of_0[189] == 0
    assert mdl.y_number_of_1[189] == 1
    assert mdl.y_number_of_0[102] == 2
    assert mdl.y_number_of_1[102] == 0
    assert mdl.y_number_of_0[52] == 0
    assert mdl.y_number_of_1[52] == 1


def test_set_training_data_warning():
    mdl = CoderNoiseCut([3, 3, 2])
    mdl.coder_set_training_data(X_train, y_train)
    with pytest.raises(Warning, match="Some data has been set before!"):
        mdl.coder_set_training_data(X_train[-1], [y_train[-1]])


def test_coder_add_training_data_no_data_exist_before():
    mdl = CoderNoiseCut([3, 3, 2])
    with pytest.raises(TypeError, match="No Data exist to add to"):
        mdl.coder_add_training_data(X_train[-1], y_train[-1])


@pytest.mark.parametrize(
    "x, y, index, n",
    [
        ([X_train[0]], y_train[0], 189, [0, 2]),
        (X_train[-1], [[y_train[-1]]], 102, [2, 0]),
    ],
)
def test_add_training_data_one_data(x, y, index, n):
    mdl = CoderNoiseCut([3, 3, 2])
    mdl.coder_set_training_data(X_train[0:-1], y_train[0:-1])

    mdl.coder_add_training_data(x, y)

    assert mdl.y_number_of_0[index] == n[0]
    assert mdl.y_number_of_1[index] == n[1]


def test_initialize_w(noisecut_model: CoderNoiseCut):
    for id_box in range(noisecut_model.n_box):
        assert not noisecut_model.all_f[id_box].w

    noisecut_model.initialize_w()

    for id_box in range(noisecut_model.n_box):
        assert type(noisecut_model.all_f[id_box].w) == numpy.ndarray


def test_set_weight_box(noisecut_model):
    dim = noisecut_model.all_f[0].n_diff_states
    noisecut_model.all_f[0].w = numpy.zeros(dim * (dim - 1) // 2, int)

    noisecut_model.set_weight_each_box_by_recursion(1)

    w_true = numpy.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            2,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
        ]
    )

    assert_array_equal(noisecut_model.all_f[0].w, w_true)

    dim = noisecut_model.all_f[1].n_diff_states
    noisecut_model.all_f[1].w = numpy.zeros(dim * (dim - 1) // 2, int)

    noisecut_model.set_weight_each_box_by_recursion(0)

    w_true = numpy.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            1,
            1,
            0,
            2,
            0,
            0,
            0,
            2,
            0,
            0,
            4,
            0,
            3,
            1,
        ]
    )

    assert_array_equal(noisecut_model.all_f[1].w, w_true)


def test_set_weight_box_more_than_one_input_for_some_states(
    noisecut_model, more_data
):
    X, y = more_data
    noisecut_model.coder_add_training_data(X, y)

    dim = noisecut_model.all_f[0].n_diff_states
    noisecut_model.all_f[0].w = numpy.zeros(dim * (dim - 1) // 2, int)

    noisecut_model.set_weight_each_box_by_recursion(1)

    w_true = numpy.array(
        [
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            2,
            1,
            0,
            2,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            2,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
        ]
    )

    assert_array_equal(noisecut_model.all_f[0].w, w_true)

    dim = noisecut_model.all_f[1].n_diff_states
    noisecut_model.all_f[1].w = numpy.zeros(dim * (dim - 1) // 2, int)

    noisecut_model.set_weight_each_box_by_recursion(0)

    w_true = numpy.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            1,
            1,
            0,
            2,
            0,
            0,
            0,
            3,
            0,
            0,
            5,
            0,
            4,
            2,
        ]
    )

    assert_array_equal(noisecut_model.all_f[1].w, w_true)


def test_count_labels(noisecut_model):
    w_true_0 = numpy.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            0,
            2,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
        ]
    )

    noisecut_model.all_f[0].w = w_true_0

    w_true_1 = numpy.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            1,
            1,
            0,
            2,
            0,
            0,
            0,
            2,
            0,
            0,
            4,
            0,
            3,
            1,
        ]
    )

    noisecut_model.all_f[1].w = w_true_1

    maxcut_solver = MaxCutSolvers()
    maxcut_solver.set_weight_1d_and_n_vertices(
        weight=noisecut_model.all_f[0].w,
        n_vertices=noisecut_model.all_f[0].n_diff_states,
    )

    noisecut_model.set_binary_function_of_box(
        0, maxcut_solver.solve_maxcut()[1]
    )

    maxcut_solver.set_weight_1d_and_n_vertices(
        weight=noisecut_model.all_f[1].w,
        n_vertices=noisecut_model.all_f[1].n_diff_states,
    )

    noisecut_model.set_binary_function_of_box(
        1, maxcut_solver.solve_maxcut()[1]
    )

    noisecut_model.count_labels_by_recursion()

    number_of_0_labels_true = numpy.array([5, 0, 12, 6])
    number_of_1_labels_true = numpy.array([2, 6, 0, 1])

    assert_array_equal(
        noisecut_model.number_of_0_labels, number_of_0_labels_true
    )
    assert_array_equal(
        noisecut_model.number_of_1_labels, number_of_1_labels_true
    )


def test_fit(training_data):
    n_each_box, X, y = training_data
    mdl = NoiseCut(n_each_box)

    mdl.fit(X, y)
    black_box_f_true = numpy.array([0, 1, 0, 0], bool)
    assert_array_equal(mdl.get_binary_function_black_box(), black_box_f_true)


def test_fit_more_data(training_data):
    n_each_box, X, y = training_data
    mdl = NoiseCut(n_each_box)

    mdl.fit(X[0:-4], y[0:-4])
    mdl.fit(X[-4:], y[-4:], with_more_data=True, print_result=True)
    black_bock_f_true = numpy.array([0, 1, 0, 0], bool)
    assert_array_equal(mdl.get_binary_function_black_box(), black_bock_f_true)


def test_performance_of_noisecut(training_data, true_data):
    n_each_box, X, y = training_data
    mdl = NoiseCut(n_each_box)

    mdl.fit(X, y)

    n_each_box, X_all, y_all = true_data

    y_predicted = mdl.predict(X_all)
    assert int(array_elements_almost_equal(y_all, y_predicted)) == 85


def test_predict_model_not_predicted():
    mdl = NoiseCut([1, 3, 2])
    with pytest.raises(RuntimeError, match="Model has not been fitted yet!"):
        mdl.predict([0, 1, 0, 1, 1, 1])


def test_predict_all(training_data, true_data):
    n_each_box, X, y = training_data
    mdl = NoiseCut(n_each_box)

    mdl.fit(X, y)

    n_each_box, X_all, y_all = true_data

    y_predicted = mdl.predict_all()
    assert int(array_elements_almost_equal(y_all, y_predicted)) == 85


def test_predict_all_model_not_predicted():
    mdl = NoiseCut([1, 3, 2])
    with pytest.raises(RuntimeError, match="Model has not been fitted yet!"):
        mdl.predict_all()


def array_elements_almost_equal(y_all, y_predicted):
    n = len(y_all)
    count = 0

    for id_y in range(n):
        if y_all[id_y] == y_predicted[id_y]:
            count += 1

    return count / n * 100


def test_initialize_w_not_initialized_before(noisecut_model):
    assert not noisecut_model.w_initialized

    noisecut_model.initialize_w()
    assert noisecut_model.w_initialized

    for i in range(noisecut_model.n_box):
        temp = 2 ** noisecut_model.n_input_each_box[i]
        expected_size = temp * (temp - 1) // 2
        assert numpy.size(noisecut_model.all_f[i].w) == expected_size


def test_initialize_w_initialized_before(noisecut_model):
    noisecut_model.initialize_w()

    for i in range(noisecut_model.n_box):
        noisecut_model.all_f[i].w[:] = 1

    noisecut_model.initialize_w()

    for i in range(noisecut_model.n_box):
        assert not sum(noisecut_model.all_f[i].w)


def test_predict_probability_of_being_1_model_not_fitted():
    mdl = NoiseCut([1, 3, 2])
    with pytest.raises(RuntimeError, match="Model has not been fitted yet!"):
        mdl.predict_probability_of_being_1([0, 1, 0, 1, 1, 1])


@pytest.mark.parametrize(
    "p, sample",
    [
        (2 / 7, [0, 0, 1, 0, 1, 0]),
        (1 / 7, [1, 0, 0, 0, 1, 1]),
        (1, [1, 1, 0, 1, 1, 1]),
        (0, [1, 0, 1, 1, 1, 0]),
    ],
)
def test_predict_probability_of_being_1_n_sample_1(
    fitted_noisecut_model, p, sample
):
    assert fitted_noisecut_model.predict_probability_of_being_1(sample) == p


@pytest.mark.parametrize(
    "p, sample",
    [
        (
            [2 / 7, 1 / 7, 1, 0],
            [
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 0, 1, 1, 1, 0],
            ],
        )
    ],
)
def test_predict_probability_of_being_1_n_sample_not_1(
    fitted_noisecut_model, p, sample
):
    assert_array_equal(
        fitted_noisecut_model.predict_probability_of_being_1(sample), p
    )


def test_set_uncertainty_measure_main(training_data):
    n_each_box, X, y = training_data
    mdl = NoiseCut(n_each_box)
    mdl.fit(X, y)
    mdl.set_uncertainty_measure()


def test_set_uncertainty_measure(training_data):
    n_each_box, X, y = training_data
    mdl = CoderNoiseCut(n_each_box)
    mdl.coder_fit(X, y)
    mdl.coder_set_uncertainty_measure(file_path_result="uncertainty")

    assert_array_equal(mdl.number_of_0_labels, [5, 0, 12, 6])
    assert_array_equal(mdl.number_of_1_labels, [2, 6, 0, 1])
    assert_array_equal(mdl.probability_of_being_1, [2 / 7, 1, 0, 1 / 7])


def test_predict_pseudo_boolean_func_coef(fitted_noisecut_model):
    assert_array_almost_equal(
        fitted_noisecut_model.predict_pseudo_boolean_func_coef(),
        [0.28571429, 0.71428571, -0.28571429, -0.57142857],
    )


@pytest.mark.parametrize(
    "s, sample",
    [
        (2, [0, 0, 1, 0, 1, 0]),
        (4, [1, 0, 0, 1, 1, 1]),
        (1, [0, 1, 1, 0, 0, 1]),
        (1, [1, 0, 1, 1, 1, 0]),
    ],
)
def test_predict_score(fitted_noisecut_model, s, sample):
    predicted_score = fitted_noisecut_model.predict_score(
        sample, vector_n_score=[0, 0.2, 0.5, 0.8, 1]
    )
    assert predicted_score == s


@pytest.mark.parametrize(
    "s, sample",
    [
        (
            [3, 5, 2, 1],
            [
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 1],
                [0, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 0],
            ],
        )
    ],
)
def test_predict_score_more_than_one_input(fitted_noisecut_model, s, sample):
    predicted_score = fitted_noisecut_model.predict_score(
        sample, vector_n_score=[0, 0.1, 0.2, 0.5, 0.8, 1]
    )
    assert_array_equal(predicted_score, s)


def test_predict_mortality_of_each_score(fitted_noisecut_model, test_data):
    x_test, y_test = test_data
    (
        mortality,
        n_0,
        n_1,
    ) = fitted_noisecut_model.predict_mortality_of_each_score(
        x_test,
        y_test,
        vector_n_score=[0, 0.2, 0.5, 0.8, 1],
        print_mortality=True,
    )

    fitted_noisecut_model.print_model()

    assert_array_almost_equal(mortality, [4.76190476, 62.5, -1, 33.33333333])
    assert_array_equal(n_0, [20, 3, 0, 2])
    assert_array_equal(n_1, [1, 5, 0, 1])


def test_print_model_while_model_not_fitted(noisecut_model):
    with pytest.raises(RuntimeError, match="Model has not been fitted yet!"):
        noisecut_model.coder_print_model()


def test_get_binary_function(fitted_noisecut_model):
    print(fitted_noisecut_model.get_binary_function_of_box(0))
    print(fitted_noisecut_model.get_binary_function_of_box(1))

    assert_array_equal(
        fitted_noisecut_model.get_binary_function_of_box(0),
        [0, 1, 0, 1, 0, 0, 1, 0],
    )
    assert_array_equal(
        fitted_noisecut_model.get_binary_function_of_box(1),
        [0, 1, 0, 1, 1, 1, 1, 0],
    )

    assert_array_equal(
        fitted_noisecut_model.get_binary_function_black_box(), [0, 1, 0, 0]
    )


def test_metric():
    y_test = [0, 0, 0, 1, 1, 0, 0]
    y_predicted = [0, 1, 0, 0, 1, 0, 1]
    assert_array_equal(
        Metric.set_confusion_matrix(y_test, y_predicted),
        [4 / 7, 1 / 2, 1 / 3, 2 / 5],
    )
