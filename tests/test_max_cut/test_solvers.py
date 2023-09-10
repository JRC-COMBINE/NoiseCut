import pytest

from noisecut.max_cut.solvers import MaxCutSolvers

from .data import Data


@pytest.fixture
def data(request):
    d = Data()
    d.initialize_data(example=request.param)
    return d


@pytest.mark.slow
@pytest.mark.parametrize("data", ["g05_60.0"], indirect=True)  # "g05_60.1"
def test_maxcut_with_local_solver(data):
    cls_maxcut = MaxCutSolvers()
    cls_maxcut.set_weight_2d_and_n_vertices(
        data.get_weight(), data.get_n_vertices()
    )
    obj, sol = cls_maxcut.solve_maxcut()
    assert int(obj) == data.true_obj
