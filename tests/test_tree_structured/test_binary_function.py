import numpy as np
import pytest
from numpy.testing import assert_array_equal

from noisecut.tree_structured.binary_function import BinaryFunction


@pytest.mark.parametrize(
    "n_binary_input",
    [1, 2, 4],
)
def test_set_random_binary_function(n_binary_input):
    mdl = BinaryFunction(n_binary_input)
    mdl.set_random_binary_function()
    assert sum(mdl.function) > 0


@pytest.mark.parametrize(
    "n_binary_input",
    [2.5, "d", [1, 2]],
)
def test_validate_n_binary_input_binary_function_is_integer(n_binary_input):
    with pytest.raises(
        Exception, match=f" should be 'integer' not {type(n_binary_input)}!"
    ):
        BinaryFunction(n_binary_input)


@pytest.mark.parametrize(
    "n_binary_input",
    [0.0, -2, -6.0],
)
def test_validate_n_binary_input_binary_function_is_greater_than_zero(
    n_binary_input,
):
    with pytest.raises(
        Exception, match=f" should be 'greater than zero', {n_binary_input}<1!"
    ):
        BinaryFunction(n_binary_input)


def test_calc_output():
    f = BinaryFunction(2)
    f.function = np.array([1, 0, 1, 0], bool)
    assert f.calc_output_func([1, 1]) == 0
    assert f.calc_output_func([0, 1]) == 1
    assert f.calc_output_func([1, 0]) == 0
    assert f.calc_output_func([0, 0]) == 1


@pytest.mark.parametrize(
    "function_a", [np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1])]
)
def test_binary_function_manually(function_a):
    f = BinaryFunction(2)
    f.set_binary_function_manually(function_a)
    assert_array_equal(function_a, f.function)


@pytest.mark.parametrize(
    "binary_func, diff_states",
    [[10, 4], [bool, 8]],
)
def test_validate_binary_input_if_is_not_array_like(binary_func, diff_states):
    with pytest.raises(
        TypeError, match="which has len, shape or __array__ attribute!"
    ):
        BinaryFunction.validate_binary_input(binary_func, diff_states)


@pytest.mark.parametrize(
    "binary_func, diff_states",
    [([[1, 2], [1]], 4), ([[1, 2], 1], 2)],
)
def test_validate_binary_input_if_is_not_homo(binary_func, diff_states):
    with pytest.raises(ValueError, match="has an inhomogeneous shape!"):
        BinaryFunction.validate_binary_input(binary_func, diff_states)


@pytest.mark.parametrize(
    "binary_func, diff_states",
    [[[[1, 0], [1, 0]], 4], [[[1], [0]], 8], [[1, 0, 1], 4]],
)
def test_validate_binary_input_if_2d_but_not_correct_size(
    binary_func, diff_states
):
    with pytest.raises(
        ValueError, match="shape of 'binary_input' is not as expected!"
    ):
        BinaryFunction.validate_binary_input(binary_func, diff_states)


def test_validate_binary_input_if_0d():
    with pytest.raises(
        ValueError, match="dimension of 'binary_input' array should be "
    ):
        BinaryFunction.validate_binary_input("01011111", 8)


@pytest.mark.parametrize(
    "binary_func, diff_states",
    [[[[3], [0], [1], [1]], 4], [["hed", 0], 2]],
)
def test_validate_binary_input_if_not_binary(binary_func, diff_states):
    with pytest.raises(
        ValueError, match="the 'binary_input' should be a 'binary' array,"
    ):
        BinaryFunction.validate_binary_input(binary_func, diff_states)


@pytest.mark.parametrize(
    "binary_func, diff_states",
    [[[[True], [0.0], [1], [1]], 4], [[True, 0], 2]],
)
def test_validate_binary_input_true(binary_func, diff_states):
    status, _ = BinaryFunction.validate_binary_input(binary_func, diff_states)
    assert status


def test_print():
    f = BinaryFunction(2)
    f.function = np.array([1, 0, 1, 0], bool)
    print(f.get_str_info_function())
    print(f)
