import pytest

from noisecut.max_cut.base_maxcut import BaseMaxCut


@pytest.fixture
def base_maxcut_class():
    cls = BaseMaxCut()
    return cls


@pytest.mark.parametrize(
    "n, size_w",
    [(2, 1), (10.0, 45)],
)
def test_validate_n_true(base_maxcut_class, n, size_w):
    assert base_maxcut_class.validate_n(n)
    assert base_maxcut_class.n_vertices == n
    assert base_maxcut_class.size_w == size_w


@pytest.mark.parametrize(
    "n",
    ["c", 10.3, [1, 2], "Hello"],
)
def test_validate_n_not_integer(base_maxcut_class, n):
    with pytest.raises(TypeError, match="not recognizable as 'integer'"):
        base_maxcut_class.validate_n(n)


@pytest.mark.parametrize(
    "n",
    [1, True, False, -1],
)
def test_validate_n_not_greater_than_one(base_maxcut_class, n):
    with pytest.raises(
        ValueError, match="Number of Nodes should be greater than one."
    ):
        base_maxcut_class.validate_n(n)


@pytest.mark.parametrize(
    "n, w",
    [(2, [1]), (4, [0.2, 0.8, 0, 1, 0.5, 0]), (3, [-0.1, 0.5, -1])],
)
def test_validate_n_w_1d_true(base_maxcut_class, n, w):
    assert base_maxcut_class.validate_n_w_1d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, 1), (4, False), (3, 3.5)],
)
def test_validate_n_w_1d_if_w_is_not_array_like(base_maxcut_class, n, w):
    with pytest.raises(TypeError, match="len, shape or __array__ attribute!"):
        base_maxcut_class.validate_n_w_1d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, [[2], 1]), (4, [[1, 2], [1]])],
)
def test_validate_n_w_1d_if_w_is_not_homo(base_maxcut_class, n, w):
    with pytest.raises(
        ValueError, match="w array has an inhomogeneous shape!"
    ):
        base_maxcut_class.validate_n_w_1d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, [0.2, -0.8]), (4, [0.2, 0.8, 0, 1, 0.5]), (2, [[0.2], [-0.8]])],
)
def test_validate_n_w_1d_if_w_dimension_is_1_not_correct_size(
    base_maxcut_class, n, w
):
    with pytest.raises(
        ValueError, match="Length of weight array is not acceptable."
    ):
        base_maxcut_class.validate_n_w_1d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, "Hello"), (4, [[[1]]])],
)
def test_validate_n_w_1d_if_w_not_correct_dimension(base_maxcut_class, n, w):
    with pytest.raises(ValueError, match="dimension of w array"):
        base_maxcut_class.validate_n_w_1d(w, n)


def test_validate_n_w_2d_true(base_maxcut_class):
    assert base_maxcut_class.validate_n_w_2d(
        [[1, 0.2, -0.5], [0.2, 1, 0.9], [-0.5, 0.9, 1]], 3
    )
    assert base_maxcut_class.weight[0] == 0.2
    assert base_maxcut_class.weight[1] == -0.5
    assert base_maxcut_class.weight[2] == 0.9


@pytest.mark.parametrize(
    "n, w",
    [(2, 1), (4, False), (3, 3.5)],
)
def test_validate_n_w_2d_not_array_like(base_maxcut_class, n, w):
    with pytest.raises(
        TypeError, match="It must have len, shape or " "__array__ attribute!"
    ):
        base_maxcut_class.validate_n_w_2d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, [[2], 1]), (4, [[1, 2], [1]])],
)
def test_validate_n_w_2d_if_w_is_not_homo(base_maxcut_class, n, w):
    with pytest.raises(
        ValueError, match="w array has an inhomogeneous shape!"
    ):
        base_maxcut_class.validate_n_w_2d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, "Hello"), (4, [[[1]]]), (3, [1, 3])],
)
def test_validate_n_w_2d_if_w_not_correct_dimension(base_maxcut_class, n, w):
    with pytest.raises(ValueError, match="dimension of w array"):
        base_maxcut_class.validate_n_w_2d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [(2, [[1], [-0.2]]), (4, [[1, 0.3], [0.3, 1]])],
)
def test_validate_n_w_2d_if_w_2d_not_square(base_maxcut_class, n, w):
    with pytest.raises(
        ValueError, match="Shape of weight array is not acceptable."
    ):
        base_maxcut_class.validate_n_w_2d(w, n)


@pytest.mark.parametrize(
    "n, w",
    [
        (3, [[1, 0.2, -1.5], [0.2, 1, 0.9], [-0.5, 0.9, 1]]),
        (2, [[1, 2], [-2, 1]]),
    ],
)
def test_validate_n_w_2d_assymmetric(base_maxcut_class, n, w):
    with pytest.raises(ValueError, match="Weight matrix in not symmetric."):
        base_maxcut_class.validate_n_w_2d(w, n)
