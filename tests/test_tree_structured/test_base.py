import numpy as np
import pandas as pd
import pytest

from noisecut.tree_structured.base import Base


@pytest.mark.parametrize(
    "input",
    [0.0, 0, True, 1],
)
def test_is_bool(input):
    assert Base.is_bool(input)


@pytest.mark.parametrize("num", [0, True, 1.0])
def test_is_bool_when_num_is_bool(num):
    assert Base.is_bool(num)


@pytest.mark.parametrize(
    "input",
    [3, 4.0, True, np.ones(3, int)[0]],
)
def test_is_integer(input):
    assert Base.is_integer(input)


@pytest.mark.parametrize(
    "input",
    [3.2, "c", "Hello", np.ones(3, int), [1, 2]],
)
def test_is_integer_false(input):
    assert not Base.is_integer(input)


@pytest.mark.parametrize(
    "input",
    [3.2, np.ones(3, float)[0], 4],
)
def test_is_float(input):
    assert Base.is_float(input)


@pytest.mark.parametrize(
    "input",
    ["c", "Hello", [1, 2], np.zeros(3)],
)
def test_is_float_false(input):
    assert not Base.is_float(input)


@pytest.mark.parametrize(
    "input",
    [[-1.1, "c"], np.array([3, 5, 3.2]), [[0]], pd.DataFrame([True, 2])],
)
def test_is_array_like(input):
    assert Base.is_array_like(input)


@pytest.mark.parametrize(
    "input",
    [3, 4.5, True],
)
def test_is_array_like_false(input):
    assert not Base.is_array_like(input)


@pytest.mark.parametrize(
    "input",
    [[True, 2, 3.0], np.zeros(2)],
)
def test_are_list_elements_integer(input):
    assert Base.are_list_elements_integer(input)


@pytest.mark.parametrize(
    "input",
    [[1, 2, "c"], [2, 2.3, 3], "r", "sbd"],
)
def test_are_list_elements_integer_false(input):
    assert not Base.are_list_elements_integer(input)


@pytest.mark.parametrize(
    "input",
    [[2, 3, 1], np.array([2, 5])],
)
def test_are_integer_elements_greater_than_zero(input):
    assert Base.are_integer_elements_greater_than_zero(input)


@pytest.mark.parametrize(
    "input",
    [[2, 3, 0], [2, -1, 3]],
)
def test_are_integer_elements_greater_than_zero_false(input):
    assert not Base.are_integer_elements_greater_than_zero(input)


@pytest.mark.parametrize(
    "input",
    [[True, 0, 1.0], np.zeros((3, 2)), pd.DataFrame([[1, 0], [0, 0]])],
)
def test_is_array_binary(input):
    assert Base.is_array_binary(input)


@pytest.mark.parametrize(
    "input",
    [
        [True, 2, 1.0],
        2 * np.ones((3, 2, 2)),
        pd.DataFrame([[1, "c"], [0, "d"]]),
    ],
)
def test_is_array_binary_false(input):
    assert not Base.is_array_binary(input)


@pytest.mark.parametrize(
    "percentage",
    ["c", "hed", [30]],
)
def test_is_percentage_is_number(percentage):
    with pytest.raises(TypeError, match="percentage should be a number"):
        Base.is_percentage(percentage)


@pytest.mark.parametrize(
    "percentage",
    ["1000", -2, 101],
)
def test_is_percentage_is_in_range(percentage):
    with pytest.raises(ValueError, match="percentage must be in range"):
        Base.is_percentage(percentage)


@pytest.mark.parametrize(
    "percentage",
    [True, 0.75, 0.2],
)
def test_is_percentage_when_get_warning(percentage):
    with pytest.raises(Warning, match="is entered on purpose."):
        Base.is_percentage(percentage)


@pytest.mark.parametrize(
    "percentage",
    [100, 0.0, 45.3],
)
def test_is_percentage_when_is_true(percentage):
    status, _ = Base.is_percentage(percentage)
    assert status
