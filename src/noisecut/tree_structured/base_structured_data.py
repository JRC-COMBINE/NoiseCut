"""Base Class for NoiseCut estimator."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>

from typing import Any

import numpy as np
import numpy.typing as npt

from noisecut.tree_structured.base import Base


class BaseStructuredData(Base):
    """
    Base class for NoiseCut estimator.

    Attributes
    ----------
    n_input_each_box : ndarray of shape (n_box,)
        An array of size `n_box` (number of first-layer black boxes) which
        keeps number of input features to each box.
        For instance, when `n_input_each_box`=[2, 4, 1], it means there are
        three first-layer black boxes and number of input features to box1,
        box2 and box3 is 2, 4, and 1, respectively.

        Note: There is only one black box in the second-layer.
    n_box : int
        Number of the first layer boxes.
    dimension : int
        Numer of input features.
    """

    def __init__(self):  # numpydoc ignore=GL08
        self.n_input_each_box: npt.NDArray[np.int_] = []
        self.n_box: int = 0
        self.dimension: int = 0

    def validate_x(self, x: Any) -> tuple[bool, npt.NDArray[np.bool_]]:
        """
        Validate input data `x`.

        Parameters
        ----------
        x : {array-like, dataframe} of shape (n_samples, n_features)

        Returns
        -------
        status : bool
            True if `x` is a valid input.
        x : ndarray of bool
            Validated input data `x` in ndarray type.

        Raises
        ------
        TypeError
            If `x` does not have has len, shape or __array__ attribute.
        ValueError
            If `x` does not have expected dimension and value
        """
        if not self.is_array_like(x):
            raise TypeError(
                f"the X should be an array which has len, shape or __array__ "
                f"attribute! Not a {type(x)}"
            )

        try:
            x = np.asarray(x)
        except ValueError:
            raise ValueError("X array has an inhomogeneous shape!")

        if len(x.shape) == 2:
            if self.dimension == 0:
                self.dimension = x.shape[1]
            elif not x.shape[1] == self.dimension:
                raise ValueError(
                    f"dimension of X array should be (n_samples, n_features="
                    f" {self.dimension}) while the value for 'n_features' is"
                    f" {x.shape[1]}"
                )
        elif len(x.shape) == 1:
            if self.dimension == 0:
                self.dimension = len(x)
            elif not len(x) == self.dimension:
                raise ValueError(
                    f"dimension of X array for 'one data' should be ("
                    f"n_features= {self.dimension},) while the input array "
                    f"has the shape {x.shape}"
                )
            x = np.asarray([x])
        else:
            raise ValueError(
                f"'dimension of X array' should be (n_samples, n_features="
                f" {self.dimension}) while the input array has the shape"
                f" {x.shape}"
            )

        if not self.is_array_binary(x):
            raise ValueError(
                "the X should be a 'binary' array, only 0 and 1 are "
                "acceptable as the elements of the array."
            )
        return True, np.asarray(x, bool)

    def validate_x_y(
        self, x: Any, y: Any
    ) -> tuple[bool, npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """
        Validate input data `x` and `y`.

        Parameters
        ----------
        x : {array-like, dataframe} of shape (n_samples, n_features)
        y : {array-like, dataframe} of shape (n_samples,)

        Returns
        -------
        status : bool
            True if `x` and `y` are valid inputs.
        x : ndarray of bool
            Validated `x` in ndarray type.
        y : ndarray of bool
            Validated `y` in ndarray type.
        """
        _, x = self.validate_x(x)
        n_samples, n_features = x.shape

        if n_samples == 1:
            if self.is_array_like(y):
                try:
                    y = np.asarray(y)
                except ValueError:
                    raise ValueError("y array has an inhomogeneous shape!")

                if not y.size == 1:
                    raise ValueError(
                        "Based on X, there should be only 'one output' as y "
                        f"while the size of y is {y.size}"
                    )
                else:
                    if not self.is_bool(y.flat[0]):
                        raise ValueError("y value should be binary (0 or 1)!")
            else:
                if not self.is_bool(y):
                    raise ValueError("y value should be binary (0 or 1)!")
                y = np.array([y])
        else:
            if not self.is_array_like(y):
                raise TypeError(
                    f"the y should be an array which has len, shape or "
                    f"__array__ attribute! Not a {type(x)}"
                )

            try:
                y = np.asarray(y)
            except ValueError:
                raise ValueError("y array has an inhomogeneous shape!")

            if len(y.shape) == 2:
                if not (y.size == n_samples and (1 in y.shape)):
                    raise ValueError(
                        f"shape of y array (n_samples,) should be in "
                        f"accordance with the shape of X array (n_samples, "
                        f"n_features) while the shape of y is {y.shape} and "
                        f"the shape of X is ({n_samples}, {n_features}) "
                    )
            elif len(y.shape) == 1:
                if not y.size == n_samples:
                    raise ValueError(
                        f"shape of y array (n_samples,) should be in "
                        f"accordance with the shape of X array (n_samples, "
                        f"n_features) while the shape of y is {y.shape} and "
                        f"the shape of X is ({n_samples}, {n_features}) "
                    )
            else:
                raise ValueError(
                    f"'dimension of y array' should be (n_samples,) while "
                    f"the input array has the shape {y.shape}"
                )

            if not self.is_array_binary(y):
                raise ValueError(
                    "the y should be a 'binary' array, only 0 and 1 are "
                    "acceptable as the elements of the array."
                )
        return True, x, y.flatten()

    def validate_n_input_each_box(self, n_input_each_box: Any) -> None:
        """
        Validate `n_input_each_box`.

        Parameters
        ----------
        n_input_each_box : {array-like, ndarray}

        Notes
        -----
        Based on the validated input, values of `n_box`, `n_input_each_box`
        and `dimension` are initialized.
        """
        if not self.is_array_like(n_input_each_box):
            raise TypeError(
                f"the class should be initialized with an array which has "
                f"len, shape or __array__ attribute! Not a"
                f" {type(n_input_each_box)}"
            )

        try:
            temp = np.asarray(n_input_each_box)
        except ValueError:
            raise ValueError(
                "'n_input_each_box' array has an inhomogeneous " "shape!"
            )

        if len(temp.shape) != 1:
            raise TypeError(
                "the class should be initialized with an array which has"
                " 1-D shape!"
            )

        if not self.are_list_elements_integer(temp):
            raise TypeError(
                "the class should be initialized with an array which has "
                "'integer' elements!"
            )
        temp = np.asarray(n_input_each_box, int)
        if not self.are_integer_elements_greater_than_zero(temp):
            raise ValueError(
                "the class should be initialized with an array which has "
                "integer elements 'greater than zero'!"
            )

        self.n_input_each_box = temp
        self.n_box = len(self.n_input_each_box)
        self.dimension = self.n_input_each_box.sum()

    def validate_id_box(self, id_box: Any) -> tuple[bool, int]:
        """
        Validate type and value of the `id_box`.

        Parameters
        ----------
        id_box : int
            Index of the box, a value between 0 and `n_box`-1.

        Returns
        -------
        status : bool
            Whether the input value for the `id_box` is valid.
        id_box : int
            Validated `id_box`.
        """
        if not self.is_integer(id_box):
            raise TypeError(
                f"Type of input {type(id_box)} for id_box is not "
                f"recognizable as 'integer'."
            )
        id_box = int(id_box)
        if not (0 <= id_box < self.n_box):
            raise ValueError(
                f"Value of id_box={id_box} is not acceptable. it should be "
                f"in range 0 to (n_box - 1)={self.n_box-1}."
            )
        return True, id_box

    def validate_vector_n_score(
        self, vector_n_score: Any
    ) -> tuple[bool, npt.NDArray[np.float_]]:
        """
        Validate `vector_n_score`.

        Parameters
        ----------
        vector_n_score : {array-like, dataframe} of shape (max_score+1,)

        Returns
        -------
        status : bool
            Whether the input value for the `vector_n_score` is valid.
        vector_n_score : ndarray of float
            Validated `vector_n_score`.
        """
        if not self.is_array_like(vector_n_score):
            raise TypeError(
                f"the 'vector_n_score' should be an array which has len, "
                f"shape or __array__ attribute! Not a type"
                f" {type(vector_n_score)}"
            )

        try:
            vector_n_score = np.asarray(vector_n_score)
        except ValueError:
            raise ValueError(
                "'vector_n_score' array has an inhomogeneous " "shape!"
            )

        if not len(vector_n_score.shape) == 1:
            raise ValueError(
                f"'vector_n_score' should be a 1D array while the input has "
                f"the shape {vector_n_score.shape} which is"
                f" {len(vector_n_score.shape)}D!"
            )

        if not vector_n_score.size > 2:
            raise ValueError(
                "Length of 'vector_n_score' should be greater than 2!"
            )

        vector_n_score.sort()
        if not (vector_n_score[0] == 0 and vector_n_score[-1] == 1):
            raise ValueError(
                "Min and Max of the `vector_n_score` should be 0 an 1, "
                "respectively. "
            )

        return True, vector_n_score

    @staticmethod
    def validate_threshold(threshold: Any) -> tuple[bool, float]:
        """
        Validate the value of threshold.

        Parameters
        ----------
        threshold : Any
            A float value in range 0 to 1, which determines the condition for
            setting binary function for the last layer black box.

        Returns
        -------
        status : bool
            Whether the input value for the `threshold` is valid.
        threshold : float
            Validated `threshold`.
        """
        if not Base.is_float(threshold):
            raise TypeError(
                f"'threshold' should be float not {type(threshold)}"
            )
        threshold = float(threshold)
        if not 0 < threshold < 1:
            raise ValueError(
                f"'threshold' should be a value in range (0,1) not"
                f" {threshold}"
            )

        if threshold != 0.5:
            raise Warning(f"Threshold has been set to {threshold}")

        return True, threshold


class BasePseudoBooleanFunc(Base):
    """
    Base class of `PseudoBooleanFunc` to validate input data.

    Attributes
    ----------
    arity : int
        Arity of the Pseudo-Boolean function.
    """

    def validate_arity(self, arity: Any) -> bool:
        """
        Validate `arity` of the Pseudo-Boolean Function.

        Parameters
        ----------
        arity : Any
            `arity` of the Pseudo-Boolean Function, should be greater than one.

        Returns
        -------
        bool
            True if `arity` is a valid input.
        """
        if not self.is_integer(arity):
            raise TypeError(
                f"Type of input {type(arity)} for arity of the "
                f"Pseudo-Boolean function is not recognizable as 'integer'."
            )
        self.arity: int = int(arity)
        if not (arity > 1):
            raise ValueError(
                f"Value of the arity of the Pseudo-Boolean function={arity} "
                f"is not acceptable. It should be greater than one."
            )
        return True

    def _validate_x(self, x: Any) -> npt.NDArray[np.int_]:
        """
        Validate `x` as an input for creating Pseudo-Boolean Function.

        Parameters
        ----------
        x : {array-like, dataframe} of shape (2**arity, arity)

        Returns
        -------
        ndarray of int
            Return `x` in the required format if it is validated.
        """
        if not self.is_array_like(x):
            raise TypeError(
                f"the x should be an array which has len, shape or __array__ "
                f"attribute! Not a {type(x)}"
            )

        try:
            x = np.asarray(x)
        except ValueError:
            raise ValueError("X array has an inhomogeneous shape!")

        if not len(x.shape) == 2:
            raise ValueError(
                f"'x array' should a 2D array while the input array is "
                f"{len(x.shape)}D!"
            )

        if not (x.shape[0] == 2 ** self.arity and x.shape[1] == self.arity):
            raise ValueError(
                f"Shape of 'x' array should be (2**arity, arity)=("
                f"{2**self.arity}, {self.arity}) while the input array has "
                f"the shape of ({x.shape[0]}, {x.shape[1]})!"
            )

        if not self.is_array_binary(x):
            raise ValueError("All elements of 'x' array should be binary!")

        numbers: list[int] = []
        for i in range(x.shape[0]):
            num: int = 0
            for j in range(x.shape[1]):
                num += x[i][j] * 2**j

            if num not in numbers:
                numbers.append(num)
            else:
                raise ValueError("The `x` binary input should be unique!")

        return x

    def validate_x_y(
        self, x: Any, y: Any
    ) -> tuple[bool, npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        Validate `x` and `y` as an input for creating Pseudo-Boolean Function.

        Parameters
        ----------
        x : {array-like, dataframe} of shape (2**arity, arity)
        y : {array-like, dataframe} of shape (2**arity,)

        Returns
        -------
        status : bool
            True if `x` and `y` are valid inputs.
        x : ndarray of bool
            Validated `x` in ndarray type.
        y : ndarray of bool
            Validated `y` in ndarray type.
        """
        x = self._validate_x(x)

        y_size: int = 2**self.arity
        if not self.is_array_like(y):
            raise TypeError(
                f"the 'y' should be an array which has len, shape or "
                f"__array__ attribute! Not a type {type(x)}"
            )

        try:
            y = np.asarray(y)
        except ValueError:
            raise ValueError("y array has an inhomogeneous shape!")

        if len(y.shape) == 1:
            if not y.size == y_size:
                raise ValueError(
                    f"Length of 'y' array should be 2**arity={y_size} "
                    f"while the 'y' array has the length of {y.size}!"
                )
        elif len(y.shape) == 2:
            if not (
                (y.shape[0] == 1 and y.shape[1] == y_size)
                or (y.shape[0] == y_size and y.shape[1] == 1)
            ):
                raise ValueError(
                    f"Shape of 'y' array is not acceptable! It should be a "
                    f"1D array of length {y_size}."
                )
        else:
            raise ValueError(
                f"Dimension of 'y' array is not acceptable! It should be a "
                f"1D array of length {y.size}."
            )
        y = y.reshape(y.size)

        return True, x, y
