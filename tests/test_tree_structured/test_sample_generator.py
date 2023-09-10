import numpy as np
import pytest

from noisecut.tree_structured.sample_generator import SampleGenerator

f1 = [0, 1, 1, 1]
f2 = [0, 0, 1, 1]
f_bb = [0, 1, 1, 0]
y_true = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0], bool)

# example without functionality
f1_w = [0, 1, 1, 1]
f2_w = [1, 1, 1, 0]
f_bb_w = [0, 1, 0, 1]


@pytest.fixture
def synthetic_data():
    example = SampleGenerator([2, 2])
    example.set_binary_function_of_box(0, f1)
    example.set_binary_function_of_box(1, f2)
    example.set_binary_function_black_box(f_bb)
    return example


@pytest.fixture
def synthetic_rand_data():
    example = SampleGenerator([2, 2], True)
    return example


@pytest.fixture
def synthetic_data_w():
    example = SampleGenerator([2, 2])
    example.set_binary_function_of_box(0, f1_w)
    example.set_binary_function_of_box(1, f2_w)
    example.set_binary_function_black_box(f_bb_w)
    return example


def test_has_functionality(synthetic_data):
    assert synthetic_data.has_synthetic_example_functionality()


def test_does_not_have_functionality(synthetic_data_w):
    assert not synthetic_data_w.has_synthetic_example_functionality()


def test_create_example():
    example = SampleGenerator([2, 2], True)
    _, _ = example.get_complete_data_set("example")
