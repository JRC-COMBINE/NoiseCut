import numpy as np
import pytest
from numpy.testing import assert_array_equal

from noisecut.tree_structured.structured_data import StructuredData

f1 = [1, 0, 1, 0]
f2 = [1, 0, 1, 1]
f_bb = [1, 1, 0, 1]
y_true = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], bool)


@pytest.fixture
def synthetic_data():
    example = StructuredData([2, 2])
    example.set_binary_function_of_box(0, f1)
    example.set_binary_function_of_box(1, f2)
    example.set_binary_function_black_box(f_bb)
    return example


def test_convert_decimal_index_to_number():
    mdl = StructuredData([3, 3, 2])
    assert mdl.convert_decimal_index_to_decimal(np.array([4, 2, 3])) == 212
    assert mdl.convert_decimal_index_to_decimal(np.array([2, 7, 0])) == 58


def test_decimal_to_reverse_binary():
    example = StructuredData([3, 2, 2])
    assert_array_equal(
        example.convert_decimal_to_reverse_binary(11, example.dimension),
        np.array([1, 1, 0, 1, 0, 0, 0], bool),
    )
    assert_array_equal(
        example.convert_decimal_to_reverse_binary(48, example.dimension),
        np.array([0, 0, 0, 0, 1, 1, 0], bool),
    )


def test_reverse_binary_to_decimal():
    example = StructuredData([3, 2, 2])
    assert 11 == example.convert_reverse_binary_to_decimal(
        np.array([1, 1, 0, 1, 0, 0, 0], bool)
    )
    assert 48 == example.convert_reverse_binary_to_decimal(
        np.array([0, 0, 0, 0, 1, 1, 0], bool)
    )


def test_build_complete_data_set_y_array(synthetic_data):
    assert_array_equal(
        synthetic_data.build_complete_data_set_y_array(), y_true
    )


def test_write_complete_data_set_in_file(synthetic_data):
    _, y = synthetic_data.get_complete_data_set("synthetic_data_01")
    assert_array_equal(y, y_true)


def test_print(synthetic_data):
    synthetic_data.print_binary_function_model()


def test_get_binary_function_of_box_true(synthetic_data):
    assert_array_equal(synthetic_data.get_binary_function_of_box(0), f1)
    assert_array_equal(synthetic_data.get_binary_function_of_box(1), f2)
    assert_array_equal(synthetic_data.get_binary_function_black_box(), f_bb)


def test_calc_output_structured_system(synthetic_data):
    input_array = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]]
    output_array = [1, 1, 0, 1]
    calc_output = synthetic_data.calc_output_structured_system_multiple_input(
        input_array
    )
    assert_array_equal(calc_output, output_array)


@pytest.mark.parametrize(
    "binary_input, decimal",
    [
        ([1, 1, 1, 0], 0),
        ([0, 1, 0, 0], 3),
        ([1, 0, 0, 1], 2),
        ([0, 0, 1, 0], 1),
    ],
)
def test_calc_decimal_input_structured_system_to_black_box(
    synthetic_data, binary_input, decimal
):
    assert (
        decimal
        == synthetic_data.calc_decimal_input_structured_system_to_black_box(
            binary_input
        )
    )
