"""MaxCut solver."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>
from typing import Union

import numpy as np
import numpy.typing as npt
from docplex.mp.model import Model

from noisecut.max_cut.base_maxcut import BaseMaxCut


class MaxCutSolvers(BaseMaxCut):
    """Maxcut solver for solving maxcut problem."""

    def set_weight_1d_and_n_vertices(
        self,
        weight: Union[npt.NDArray[np.float_], list[float]],
        n_vertices: int,
    ) -> None:
        """
        Set Number of vertices and weight of edges between vertices.

        Parameters
        ----------
        weight : ndarray of shape(n_edges,)
            1D array weight data, see Notes.
        n_vertices : int
            Number of vertices.

        Notes
        -----
        When n_vertices=4, the weight data can be stored in two different
        formats:
            2D array : w_2d[node A][node B]
            1D array : w_1d[index of array] = w_2d[node_A][node_B]
                Index of 1D array:(node A, node B) -> 0:(0,1), 1:(0,2),
                2:(0,3), 3:(1,2), 4:(1,3), 5:(2,3)

                Index of array : int
                    node_A * {2 * n_vertices - 3 - node_A} // 2 + node_B - 1
        """
        assert self.validate_n_w_1d(weight, n_vertices)

    def set_weight_2d_and_n_vertices(
        self,
        weight: Union[npt.NDArray[np.float_], list[float]],
        n_vertices: int,
    ):
        """
        Set Number of vertices and weight of edges between vertices.

        Parameters
        ----------
        weight : ndarray of shape(n_vertices, n_vertices)
            2D array weight data, see Notes.

        n_vertices : int
            Number of vertices.

        Notes
        -----
        When n_vertices=4, the weight data can be stored in two different
        formats:
            2D array : w_2d[node A][node B]
            1D array : w_1d[index of array] = w_2d[node_A][node_B]
                Index of 1D array:(node A, node B) -> 0:(0,1), 1:(0,2),
                2:(0,3), 3:(1,2), 4:(1,3), 5:(2,3)

                Index of array : int
                    node_A * {2 * n_vertices - 3 - node_A} // 2 + node_B - 1
        """
        assert self.validate_n_w_2d(weight, n_vertices)

    def calc_index_of_linear_array(self, i: int, j: int) -> int:
        """
        Return index of `weight` array between two vertices of i and j.

        Parameters
        ----------
        i : int
            Number representative of a specific vertex.
        j : int
            Number representative of a specific vertex.

        Returns
        -------
        int
            Index of `weight` array between two vertices of i and j.

        Notes
        -----
        When n_vertices=4, the weight data can be stored in two different
        formats:
            2D array : w_2d[node A][node B]
            1D array : w_1d[index of array] = w_2d[node_A][node_B]
                Index of 1D array:(node A, node B) -> 0:(0,1), 1:(0,2),
                2:(0,3), 3:(1,2), 4:(1,3), 5:(2,3)

                Index of array : int
                    node_A * {2 * n_vertices - 3 - node_A} // 2 + node_B - 1
        """
        return i * (2 * self.n_vertices - 3 - i) // 2 + j - 1

    def solve_maxcut(self) -> tuple[float, npt.NDArray[np.bool_]]:
        """
        Solve MaxCut problem by utilizing CPLEX optimization package.

        Returns
        -------
        objective : float
            Value of the maximum cut.
        solution : ndarray of bool of shape(n_vertices,)
            Label vertices to 0 or 1 to divide the set to two sets. Index of
            the `sol` array is representative of each vertex, like `sol[0]`
            represents the label for the vertex 0.
        """
        sol = np.zeros(self.n_vertices, bool)
        mdl = Model(name="MaxCut")
        x = mdl.binary_var_list(keys=self.n_vertices, name="x")

        obj = mdl.sum(
            self.weight[self.calc_index_of_linear_array(i, j)]
            * (x[i] - x[j])
            * (x[i] - x[j])
            for i in range(self.n_vertices)
            for j in range(i + 1, self.n_vertices)
        )

        mdl.maximize(obj)

        assert mdl.solve(), "Solve failed"

        for i in range(self.n_vertices):
            sol[i] = round(x[i].solution_value, 0)

        return mdl.objective_value, sol
