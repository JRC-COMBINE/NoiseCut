"""Base for MaxCut solvers."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>

from typing import (
    Any,
    Union,
)

import numpy as np
import numpy.typing as npt
import scipy

from noisecut.tree_structured.base import Base


class BaseMaxCut(Base):
    """
    Base for Maxcut Solver to validate inputs.

    Attributes
    ----------
    n_vertices : int
        Number of vertices of the graph.
    size_w : int
        Size of `weight` array.
    weight : ndarray of float of shape (`size_w`,)
        Weight of edge between each two vertices (see Notes).

    Notes
    -----
    The `weight` is stored for a simple example when `n_vertices`=4 as
    below:
        weight[index of array] = w_2d[node_A][node_B]
            Index of 1D array:(node A, node B) -> 0:(0,1), 1:(0,2),
            2:(0,3), 3:(1,2), 4:(1,3), 5:(2,3)

            Index of array : int
                node_A * {2 * n_vertices - 3 - node_A} // 2 + node_B - 1
    """

    def __init__(self):
        self.n_vertices: int = 0
        self.size_w: int = 0
        self.weight: Union[None, npt.NDArray[np.float_]] = None

    def validate_n(self, n: Any) -> bool:
        """
        Validate number of vertices `n`.

        Parameters
        ----------
        n : int
            Number of vertices.

        Returns
        -------
        bool
            True if `n` is a valid input.
        """
        if not self.is_integer(n):
            raise TypeError(
                f"Type of input {type(n)} for Number of nodes is not "
                f"recognizable as 'integer' by MaxCutSolvers(). "
            )
        n = int(n)
        if not n > 1:
            raise ValueError(
                f"Number of Nodes {n} is not acceptable by MaxCutSolvers(). "
                f"Number of Nodes should be greater than one."
            )

        self.n_vertices = n
        self.size_w = n * (n - 1) // 2
        self.weight = None
        return True

    def validate_n_w_1d(self, w: Any, n: Any) -> bool:
        """
        Validate number of vertices and weight array.

        Parameters
        ----------
        w : {array-like, ndarray, dataframe} of shape (size_w,)
            1D array weight data, see Notes.
        n : int
            Number of vertices.

        Returns
        -------
        bool
            True if `w` and 'n' are valid inputs.

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
        self.validate_n(n)

        if not self.is_array_like(w):
            raise TypeError(
                f"Type of input {type(w)} for weight is not recognizable as "
                f"an array by MaxCutSolvers(). It must have len, shape or "
                f"__array__ attribute!"
            )

        try:
            w = np.asarray(w)
        except ValueError:
            raise ValueError("w array has an inhomogeneous shape!")

        if len(w.shape) == 1:
            if not len(w) == self.size_w:
                raise ValueError(
                    f"Length of weight array is not acceptable. "
                    f"MaxCutSolvers() expects an array of length n_nodes*("
                    f"n_nodes-1)/2 = {self.size_w}"
                )
        elif len(w.shape) == 2 and (1 in w.shape):
            if not w.size == self.size_w:
                raise ValueError(
                    f"Length of weight array is not acceptable. "
                    f"MaxCutSolvers() expects an array of length n_nodes*("
                    f"n_nodes-1)/2 = {self.size_w}"
                )
        else:
            raise ValueError(
                f"'dimension of w array' should be (n_nodes*[n_nodes-1]/2 "
                f"={self.size_w},) while the input array has the shape"
                f" {w.shape}"
            )
        self.weight = w.flatten()
        return True

    def validate_n_w_2d(self, w: Any, n: Any) -> bool:
        """
        Validate number of vertices and weight array.

        Parameters
        ----------
        w : {array-like, ndarray, dataframe} of shape (n_vertices, n_vertices)
            2D array weight data, see Notes.

        n : int
            Number of vertices.

        Returns
        -------
        bool
            True if `w` and 'n' are valid inputs.

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
        self.validate_n(n)

        if not self.is_array_like(w):
            raise TypeError(
                f"Type of input {type(w)} for weight is not recognizable as "
                f"an array by MaxCutSolvers(). It must have len, shape or "
                f"__array__ attribute!"
            )

        try:
            w = np.asarray(w)
        except ValueError:
            raise ValueError("w array has an inhomogeneous shape!")

        if len(w.shape) == 2:
            if (not w.shape[0] == w.shape[1]) or (
                not w.shape[0] == self.n_vertices
            ):
                raise ValueError(
                    f"Shape of weight array is not acceptable. "
                    f"MaxCutSolvers() expects an array of shape (n_nodes, "
                    f"n_nodes)=({self.n_vertices},{self.n_vertices}) not"
                    f" {w.shape}"
                )
        else:
            raise ValueError(
                f"'dimension of w array' should be (n_nodes, n_nodes)="
                f"({self.n_vertices},{self.n_vertices}) while the input "
                f"array has the shape {w.shape}"
            )

        if not scipy.linalg.issymmetric(w):
            raise ValueError("Weight matrix in not symmetric.")

        self.weight = np.zeros(self.size_w, float)
        index: int = 0
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                self.weight[index] = w[i][j]
                index += 1

        return True
