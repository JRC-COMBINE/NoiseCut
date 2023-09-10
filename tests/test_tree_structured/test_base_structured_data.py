import numpy as np
import pytest
from numpy.testing import assert_array_equal

from noisecut.model.noisecut_coder import PseudoBooleanFunc
from noisecut.tree_structured.base_structured_data import BaseStructuredData


@pytest.fixture
def structured_base_class():
    cls = BaseStructuredData()
    cls.validate_n_input_each_box([2, 3])
    return cls


@pytest.fixture
def pseudo_boolean_func_class():
    cls = PseudoBooleanFunc(2)
    return cls


@pytest.mark.parametrize(
    "n_value", [True, 4.5, 2], ids=["bool", "float", "int"]
)
def test_initialization_n_input_each_box_is_array_like(n_value):
    with pytest.raises(
        TypeError, match="which has len, shape or __array__ attribute!"
    ):
        BaseStructuredData().validate_n_input_each_box(n_value)


def test_initialization_n_input_each_box_is_not_homo():
    with pytest.raises(ValueError, match="array has an inhomogeneous shape!"):
        BaseStructuredData().validate_n_input_each_box([[2, 3, 5], [1]])


def test_initialization_n_input_each_box_dimension():
    with pytest.raises(TypeError, match="which has 1-D shape!"):
        BaseStructuredData().validate_n_input_each_box([[[2], [3]]])


@pytest.mark.parametrize("n_value", [3.5, "h"], ids=["float", "char"])
def test_initialization_n_input_each_box_is_integer(n_value):
    with pytest.raises(TypeError, match="which has 'integer' elements!"):
        BaseStructuredData().validate_n_input_each_box([n_value, 2, 3])


@pytest.mark.parametrize(
    "n_value",
    [-2, 0, False],
)
def test_initialization_n_input_each_box_greater_than_zero(n_value):
    with pytest.raises(
        ValueError, match="which has integer elements 'greater than zero'!"
    ):
        BaseStructuredData().validate_n_input_each_box([n_value, 2, 3])


def test_initialization_n_input_each_box_pass_if_no_error():
    mdl = BaseStructuredData()
    mdl.validate_n_input_each_box([2.0, 3, 3])
    assert mdl.dimension == 8
    assert mdl.n_box == 3


@pytest.mark.parametrize(
    "x",
    [[[1, 2], [1]]],
)
def test_validate_x_if_is_not_homo(structured_base_class, x):
    with pytest.raises(ValueError, match="X array has an inhomogeneous shape"):
        structured_base_class.validate_x(x)


@pytest.mark.parametrize(
    "x",
    [2, True],  # [[0, 1, 0, 1, 0]],
)
def test_validate_x_if_is_not_array_like(structured_base_class, x):
    with pytest.raises(
        TypeError,
        match="the X should be an array "
        "which has len, shape or __array__ attribute!"
        f" Not a {type(x)}",
    ):
        structured_base_class.validate_x(x)


def test_validate_x_if_shape_is_2d_1d_and_not_match_n_features(
    structured_base_class,
):
    with pytest.raises(
        ValueError, match="while the value for 'n_features' is 4"
    ):
        structured_base_class.validate_x([[1, 3, 4, 5], [0, 1, 7, 8]])

    with pytest.raises(
        ValueError, match="dimension of X array for 'one data' should be"
    ):
        structured_base_class.validate_x([1, 3, 4, 5])


@pytest.mark.parametrize(
    "x",
    ["hed", [[[1, 3, 4, 5], [0, 1, 7, 8]]]],
)
def test_validate_x_if_shape_is_not_2d_1d(structured_base_class, x):
    with pytest.raises(ValueError, match="'dimension of X array' should be"):
        structured_base_class.validate_x(x)


def test_validate_x_if_array_is_not_binary(structured_base_class):
    with pytest.raises(ValueError, match="the X should be a 'binary' array, "):
        structured_base_class.validate_x([[1, 3, 4, 5, 6], [0, 1, 7, 8, 7]])


@pytest.mark.parametrize(
    "x",
    [[[0, 1, 0, 1, 0]], [[0, 1, 0, 1, 0], [1, 1, 1, 0, 0]]],
)
def test_validate_x_true(structured_base_class, x):
    status, _ = structured_base_class.validate_x(x)
    assert status


@pytest.mark.parametrize(
    "x, y",
    [
        ([[0, 1, 0, 1, 0]], np.array([[0]])),
        ([[0, 1, 1, 0, 0], [1, 0, 1, 0, 0]], [1, 0]),
    ],
)
def test_validate_x_y_true(structured_base_class, x, y):
    status, _, _ = structured_base_class.validate_x_y(x, y)
    assert status


@pytest.mark.parametrize(
    "x, y",
    [([[0, 1, 0, 1, 0]], [[1, 0], [1]]), ([[0, 1, 1, 0, 0]], [1, [0, 1]])],
)
def test_validate_x_y_if_n_sample_1_and_y_array_not_homo(
    structured_base_class, x, y
):
    with pytest.raises(ValueError, match="y array has an inhomogeneous shape"):
        structured_base_class.validate_x_y(x, y)


@pytest.mark.parametrize(
    "x, y",
    [([[0, 1, 0, 1, 0]], [1, 0]), ([[0, 1, 1, 0, 0]], [])],
)
def test_validate_x_y_if_n_sample_1_and_y_array_like_but_y_size_not_1(
    structured_base_class, x, y
):
    with pytest.raises(
        ValueError, match="Based on X, there should be only 'one output' as y "
    ):
        structured_base_class.validate_x_y(x, y)


def test_validate_x_y_if_n_sample_1_and_y_not_array_like(
    structured_base_class,
):
    status, _, _ = structured_base_class.validate_x_y([[0, 1, 0, 1, 0]], 1)
    assert status


@pytest.mark.parametrize(
    "x, y",
    [([[0, 1, 0, 1, 0]], 5), ([[0, 1, 1, 0, 0]], "h")],
)
def test_validate_x_y_if_n_sample_1_but_y_not_binary(
    structured_base_class, x, y
):
    with pytest.raises(ValueError, match="y value should be binary"):
        structured_base_class.validate_x_y(x, y)


@pytest.mark.parametrize(
    "x, y",
    [
        ([[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]], 1),
        ([[0, 1, 1, 0, 0], [0, 1, 0, 1, 1]], 3.4),
    ],
)
def test_validate_x_y_if_n_sample_not_1_and_y_not_array_like(
    structured_base_class, x, y
):
    with pytest.raises(
        TypeError,
        match="the y should be an array "
        "which has len, shape or __array__ attribute!",
    ):
        structured_base_class.validate_x_y(x, y)


@pytest.mark.parametrize(
    "x, y",
    [
        ([[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]], [1, [0, 1]]),
        ([[0, 1, 1, 0, 0], [0, 1, 0, 1, 1]], [[1, 0, 1], [0, 1]]),
    ],
)
def test_validate_x_y_if_n_sample_not_1_and_y_not_homo(
    structured_base_class, x, y
):
    with pytest.raises(ValueError, match="y array has an inhomogeneous shape"):
        structured_base_class.validate_x_y(x, y)


@pytest.mark.parametrize(
    "x, y",
    [
        ([[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]], [1]),
        ([[0, 1, 1, 0, 0], [0, 1, 0, 1, 1]], [[1, 0, 1]]),
    ],
)
def test_validate_x_y_if_n_sample_not_1_and_n_samples_differ_in_x_y(
    structured_base_class, x, y
):
    with pytest.raises(
        ValueError, match="should be in accordance with the shape of X array"
    ):
        structured_base_class.validate_x_y(x, y)


def test_validate_x_y_if_n_sample_not_1_and_y_dimension_not_acceptable(
    structured_base_class,
):
    with pytest.raises(ValueError, match="'dimension of y array' should be "):
        structured_base_class.validate_x_y(
            [[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]], [[[0, 1]]]
        )


@pytest.mark.parametrize(
    "x, y",
    [
        ([[0, 1, 0, 1, 0], [1, 1, 0, 0, 0]], [4, 0]),
        ([[0, 1, 1, 0, 0], [0, 1, 0, 1, 1]], [["d", 0]]),
    ],
)
def test_validate_x_y_if_n_sample_not_1_and_y_not_binary(
    structured_base_class, x, y
):
    with pytest.raises(ValueError, match="the y should be a 'binary' array, "):
        structured_base_class.validate_x_y(x, y)


@pytest.mark.parametrize(
    "id_box", [[1], 4.5, "1"], ids=["list", "float", "str"]
)
def test_validate_id_box_not_integer(structured_base_class, id_box):
    with pytest.raises(
        TypeError,
        match=f"Type of input {type(id_box)} for id_box is "
        f"not recognizable as 'integer'.",
    ):
        structured_base_class.validate_id_box(id_box)


@pytest.mark.parametrize(
    "id_box", [-1, 2], ids=["negative", "greater than n_box-1"]
)
def test_validate_id_box_value_not_acceptable(structured_base_class, id_box):
    with pytest.raises(
        ValueError,
        match=f"Value of id_box={id_box} is not acceptable. "
        f"it should be in range 0 to ",
    ):
        structured_base_class.validate_id_box(id_box)


@pytest.mark.parametrize(
    "id_box",
    [0.0, 1, True],
)
def test_validate_id_box_true(structured_base_class, id_box):
    status, _ = structured_base_class.validate_id_box(id_box)
    assert status


@pytest.mark.parametrize(
    "threshold", ["h", "hedi", [1, 2]], ids=["char", "str", "list"]
)
def test_validate_threshold_not_float(threshold):
    with pytest.raises(TypeError, match="'threshold' should be float"):
        BaseStructuredData.validate_threshold(threshold)


def test_validate_threshold_warning():
    with pytest.raises(Warning, match=f"Threshold has been set to {0.6}"):
        BaseStructuredData.validate_threshold(0.6)


@pytest.mark.parametrize(
    "threshold", [0, 1, 4.5, -1], ids=["0", "1", "4.0", "-1.1"]
)
def test_validate_threshold_incorrect_range(threshold):
    with pytest.raises(ValueError, match="should be a value in range"):
        BaseStructuredData.validate_threshold(threshold)


def test_validate_threshold_true():
    assert BaseStructuredData.validate_threshold(0.5)


@pytest.mark.parametrize(
    "arity",
    [2, 5, 3.0],
)
def test_pseudo_validate_true(arity):
    PseudoBooleanFunc(arity)


@pytest.mark.parametrize(
    "arity", [[1], 3.5, "1"], ids=["list", "float", "str"]
)
def test_pseudo_validate_arity_not_integer(arity):
    with pytest.raises(TypeError, match="is not recognizable as 'integer'"):
        PseudoBooleanFunc(arity)


@pytest.mark.parametrize("arity", [-1, 1], ids=["negative", "one"])
def test_pseudo_validate_arity_not_greater_than_one(arity):
    with pytest.raises(ValueError, match="It should be greater than one."):
        PseudoBooleanFunc(arity)


@pytest.mark.parametrize("x", [2, 4.5, True], ids=["int", "float", "bool"])
def test_pseudo_validate_x_y_when_x_is_not_array(pseudo_boolean_func_class, x):
    with pytest.raises(
        TypeError, match="x should be an array which has " "len,"
    ):
        pseudo_boolean_func_class.get_coef_boolean_func(x, [1, 0, 0, 1])


def test_pseudo_validate_x_y_when_x_is_not_homogeneous(
    pseudo_boolean_func_class,
):
    x = [[1, 0], [1, 0, 1]]
    with pytest.raises(ValueError, match="X array has an inhomogeneous shape"):
        pseudo_boolean_func_class.get_coef_boolean_func(x, [1, 0, 0, 1])


@pytest.mark.parametrize(
    "x", ["hedi", "h", [[[0], [1]]]], ids=["str", "char", "3D"]
)
def test_pseudo_validate_x_y_when_x_shape_is_not_2d(
    pseudo_boolean_func_class, x
):
    with pytest.raises(ValueError, match="'x array' should a 2D array"):
        pseudo_boolean_func_class.get_coef_boolean_func(x, [1, 0, 0, 1])


@pytest.mark.parametrize(
    "x",
    [
        [[0, 0], [1, 0], [0, 1]],
        [[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1]],
    ],
    ids=["incorrect row", "incorrect column"],
)
def test_pseudo_validate_x_y_when_x_shape_is_not_acceptable(
    pseudo_boolean_func_class, x
):
    with pytest.raises(ValueError, match="Shape of 'x' array"):
        pseudo_boolean_func_class.get_coef_boolean_func(x, [1, 0, 0, 1])


@pytest.mark.parametrize(
    "x",
    [
        [[0, 2], [1, 0], [0, 1], [1, 1]],
        [[0, 1], [1, 0], [4.5, 1], [1, 1]],
    ],
    ids=["2", "4.5"],
)
def test_pseudo_validate_x_y_when_x_array_is_not_binary(
    pseudo_boolean_func_class, x
):
    with pytest.raises(ValueError, match="array should be binary"):
        pseudo_boolean_func_class.get_coef_boolean_func(x, [1, 0, 0, 1])


@pytest.mark.parametrize("y", [2, 4.5, True], ids=["int", "float", "bool"])
def test_pseudo_validate_x_y_when_y_is_not_array(pseudo_boolean_func_class, y):
    with pytest.raises(TypeError, match="'y' should be an array"):
        pseudo_boolean_func_class.get_coef_boolean_func(
            [[0, 0], [1, 0], [0, 1], [1, 1]], y
        )


def test_pseudo_validate_x_y_when_y_is_not_homo(pseudo_boolean_func_class):
    with pytest.raises(ValueError, match="y array has an inhomogeneous shape"):
        pseudo_boolean_func_class.get_coef_boolean_func(
            [[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1], [1]]
        )


@pytest.mark.parametrize(
    "y", [[0, 1, 0], [0, 1, 0, 1, 1]], ids=["less length", "high length"]
)
def test_pseudo_validate_x_y_when_length_y_is_not_correct(
    pseudo_boolean_func_class, y
):
    with pytest.raises(ValueError, match="Length of 'y' array should be"):
        pseudo_boolean_func_class.get_coef_boolean_func(
            [[0, 0], [1, 0], [0, 1], [1, 1]], y
        )


@pytest.mark.parametrize(
    "y",
    [[[1, 0], [0, 1]], [[1, 0, 1]], [[1], [0], [1], [0], [1]]],
    ids=["(2,2)", "(1,3)", "(5,1)"],
)
def test_pseudo_validate_x_y_when_shape_y_is_not_correct(
    pseudo_boolean_func_class, y
):
    with pytest.raises(ValueError, match="Shape of 'y' array is not accept"):
        pseudo_boolean_func_class.get_coef_boolean_func(
            [[0, 0], [1, 0], [0, 1], [1, 1]], y
        )


@pytest.mark.parametrize(
    "y", ["hed", "h", [[[1, 0, 1, 0]]]], ids=["str", "char", "3D"]
)
def test_pseudo_validate_x_y_when_dimension_y_is_not_correct(
    pseudo_boolean_func_class, y
):
    with pytest.raises(ValueError, match="Dimension of 'y' array"):
        pseudo_boolean_func_class.get_coef_boolean_func(
            [[0, 0], [1, 0], [0, 1], [1, 1]], y
        )


def test_pseudo_validate_x_y_when_x_rows_are_not_unique(
    pseudo_boolean_func_class,
):
    x = [[0, 0], [0, 1], [0, 0], [1, 1]]
    with pytest.raises(ValueError, match="`x` binary input should be unique"):
        pseudo_boolean_func_class.validate_x_y(x, [1, 0, 0, 1])


def test_pseudo_validate_x_y_true(pseudo_boolean_func_class):
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    status, _, _ = pseudo_boolean_func_class.validate_x_y(x, [1, 0, 0, 1])
    assert status


@pytest.mark.parametrize(
    "vector_n_score", [2, 4.5, True], ids=["int", "float", "bool"]
)
def test_validate_vector_n_score_when_is_not_array(
    structured_base_class, vector_n_score
):
    with pytest.raises(TypeError, match="'vector_n_score' should be an array"):
        structured_base_class.validate_vector_n_score(vector_n_score)


def test_validate_vector_n_score_when_is_not_homo(structured_base_class):
    with pytest.raises(ValueError, match="inhomogeneous shape"):
        structured_base_class.validate_vector_n_score([[0, 0.2], [1]])


@pytest.mark.parametrize(
    "vector_n_score",
    ["h", "hed", [[0, 0.2, 0.6, 1]]],
    ids=["char", "str", "2D"],
)
def test_validate_vector_n_score_when_is_not_1d(
    structured_base_class, vector_n_score
):
    with pytest.raises(ValueError, match="should be a 1D"):
        structured_base_class.validate_vector_n_score(vector_n_score)


def test_validate_vector_n_score_when_length_is_less_than_3(
    structured_base_class,
):
    with pytest.raises(ValueError, match="Length of 'vector_n_score'"):
        structured_base_class.validate_vector_n_score([0, 1])


@pytest.mark.parametrize(
    "vector_n_score",
    [[1, 3, 6, 8], [-1, 2, 4, 5], [0, 0.2, 1.1, 0.6, 1]],
    ids=["[1, 3, 6, 8]", "-1", "unsorted+1.1"],
)
def test_validate_vector_n_score_when_does_not_have_min_max(
    structured_base_class, vector_n_score
):
    with pytest.raises(ValueError, match="Min and Max"):
        structured_base_class.validate_vector_n_score(vector_n_score)


def test_validate_vector_n_score_unsorted(structured_base_class):
    status, v = structured_base_class.validate_vector_n_score([0.2, 1, 0.3, 0])
    assert_array_equal(v, np.array([0, 0.2, 0.3, 1]))
    assert status
