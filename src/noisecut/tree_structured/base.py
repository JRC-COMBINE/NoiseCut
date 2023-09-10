"""Base Class for checking data type."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>

from typing import (
    Any,
    Union,
)

import numpy as np
import numpy.typing as npt


class Base:
    """Validate basic types of the variables and their expected values."""

    @staticmethod
    def is_bool(num: Any) -> bool:
        """
        Return whether `num` is bool.

        Parameters
        ----------
        num : Any
            Input value to be validated as bool.

        Returns
        -------
        bool
            True if `num` is boolean {0,1}.
        """
        if num in [True, False]:
            return True
        return False

    @staticmethod
    def is_integer(num: Any) -> bool:
        """
        Return whether the `num` is integer.

        Parameters
        ----------
        num : Any
            Input value to be validated as int.

        Returns
        -------
        bool
            True if `num` is integer and False if `num` is not an integer.
        """
        if isinstance(num, int):
            return True
        try:
            if num.is_integer():
                return True
        except AttributeError:
            return False

        return False

    @staticmethod
    def is_float(num: Any) -> bool:
        """
        Return whether the `num` is float.

        Parameters
        ----------
        num : Any
            Input value to be validated as float.

        Returns
        -------
        bool
            True if `num` is float and False if `num` is not float.
        """
        try:
            float(num)
        except:  # noqa: E722
            return False
        return True

    @staticmethod
    def are_list_elements_integer(array: Union[list, npt.NDArray]) -> bool:
        """
        Return whether all elements of an `array` are integer.

        Parameters
        ----------
        array : {array-like}

        Returns
        -------
        bool
            True if all elements of `array` are integer and False if this
            condition is not fulfilled.
        """
        array_flat = np.asarray(array).flat

        for num in array_flat:
            if not Base.is_integer(num):
                return False
        return True

    @staticmethod
    def are_integer_elements_greater_than_zero(
        array: Union[list[int], npt.NDArray[np.int_]],
    ) -> bool:
        """
        Return whether all elements of `array` are greater than zero.

        Parameters
        ----------
        array : list or ndarray of int
            Input list to be validated if all its elements are greater than
            zero.

        Returns
        -------
        bool
            True if all elements of `array` are greater than zero and False
            if this condition is not fulfilled.
        """
        array_flat = np.asarray(array).flat

        for num in array_flat:
            if num < 1:
                return False

        return True

    @staticmethod
    def is_array_like(x: Any) -> bool:
        """
        Return whether `x` is in format of an array.

        Parameters
        ----------
        x : Any
            Input value to be validated as an array.

        Returns
        -------
        bool
            True if `x` in like an array and False if it is not.
        """
        return (
            hasattr(x, "__len__")
            or hasattr(x, "shape")
            or hasattr(x, "__array__")
        )

    @staticmethod
    def is_array_binary(array: Union[list, npt.NDArray]) -> bool:
        """
        Return whether all the elements of `array` are boolean.

        Parameters
        ----------
        array : list or ndarray
            Input array to be validated if all its elements are binary.

        Returns
        -------
        bool
            True if all `array` elements are boolean.
        """
        array_flat: np.flatiter = np.asarray(array).flat

        for elem in array_flat:
            if not Base.is_bool(elem):
                return False
        return True

    @staticmethod
    def is_percentage(percent: Any) -> tuple[bool, float]:
        """
        Return Whether `percent` is a number in range 0 to 100.

        Parameters
        ----------
        percent : {float, int}

        Returns
        -------
        status : bool
            True if `percent` is a valid input.
        percent : float
            Validated `percent` in float type.
        """
        if not Base.is_float(percent):
            raise TypeError(
                f"Input value for the percentage should be a number not"
                f" {type(percent)}"
            )

        percent = float(percent)
        if not 0 <= percent <= 100:
            raise ValueError(
                f"Value of the percentage must be in range [0,100]. "
                f"The input value is {percent}"
            )

        if percent <= 1 and percent != 0:
            raise Warning(
                f"Value of the percentage must be in range [0,100]. Please "
                f"make sure that the input value for percent, "
                f"which is {percent}, is entered on purpose."
            )

        return True, percent
