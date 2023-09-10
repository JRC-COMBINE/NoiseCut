"""
Binary Function.
"""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>

import random
from typing import (
    Any,
    Union,
)

import numpy as np
import numpy.typing as npt

from noisecut.tree_structured.base import Base


class BinaryFunction:
    """
    BinaryFunction class helps to define a binary function and work with.

    Parameters
    ----------
    n_binary_input : int
        It should be greater than zero. `n_binary_input` value is number
        of input features to a binary function.

    Attributes
    ----------
    n_diff_states : int
        Number of possible unique arrangements of the binary function.
        For instance, if `n_binary_input=2`, then there are 2**2=4
        possible unique arrangements for this function. [0,0], [1,0],
        [0,1], [1,1].

    function : ndarray of bool of size (n_diff_states,)
        For each unique arrangement of the binary input, `function`
        store one binary value as a response of the function to that
        input arrangement. Also, the binary values are stored in an
        order.

        For instance, if `n_binary_input=2`,

        `function[0]` stores the boolean response of the function to
        [0, 0] as (2**0)* 0 + (2**1)* 0 = 0.

        `function[1]` stores the boolean response of the function to
        [1, 0] as (2**0)* 1 + (2**1)* 0 = 1.

        `function[2]` stores the boolean response of the function to
        [0, 1] as (2**0)* 0 + (2**1)* 1 = 2.

        `function[3]` store the boolean response of the function to
        [1, 1] as (2**0)* 1 + (2**1)* 1 = 3.
    w : ndarray of size (), default = None
        In NoiseCut model each box has a binary function. This array
        is used for storing calculated weights of the function in
        NoiseCut model.
    """

    def __init__(self, n_binary_input: int) -> None:
        self.validate_n_binary_input_binary_function(n_binary_input)

        self.n_binary_input: int = int(n_binary_input)
        self.n_diff_states: int = 2**self.n_binary_input
        self.function: npt.NDArray[np.bool_] = np.zeros(
            self.n_diff_states, bool
        )

        self.w: Union[npt.NDArray[np.int_], None] = None

    def set_random_binary_function(self) -> None:
        """Set binary `function` values randomly."""
        if self.n_binary_input == 1:
            self.function[0] = False
            self.function[1] = True
        else:
            for i in range(self.n_diff_states):
                self.function[i] = random.randint(0, 1)
            if sum(self.function) == 0:
                self.set_random_binary_function()

    def set_binary_function_manually(
        self,
        input_function: Union[list[bool], npt.NDArray[np.bool_]],
    ) -> None:
        """
        Set binary `function` values by the `input_function`.

        Parameters
        ----------
        input_function : {array_like, ndarray} of size (n_diff_states,)
            Input binary function value to set `function` with it.
        """
        _, self.function = self.validate_binary_input(
            input_function, self.n_diff_states
        )

    def calc_output_func(
        self,
        binary_input: Union[list[bool], npt.NDArray[np.bool_]],
    ) -> np.bool_:
        """
        Return calculated output of the function based on the binary input.

        Parameters
        ----------
        binary_input : ndarray of size `n_binary_input`
            Input array to the binary `function`.

        Returns
        -------
        bool
            Output of function for `binary_input` as input.
        """
        _, validated_binary_input = self.validate_binary_input(
            binary_input, self.n_binary_input
        )

        index: int = 0
        for i in range(self.n_binary_input):
            index += 2**i * validated_binary_input[i]
        return self.function[index]

    def __repr__(self) -> str:
        return self.get_str_info_function()

    def __convert_decimal_to_reverse_binary(
        self, decimal: int
    ) -> npt.NDArray[np.bool_]:
        """
        Convert `decimal` to reverse_binary.

        Parameters
        ----------
        decimal : int
            See Notes section below.

        Returns
        -------
        ndarray of bool of shape(n_binary_input, )
            Return reverse binary of the decimal value.

        Notes
        -----
        to understand the definition of the terms better let's consider,
        `n_binary_input=4`:
            reverse_binary : ndarray of shape (dimension,)
                binary representation of a value in which first element in
                the array has the least value. For instance,
                `reverse_binary` representation of `decimal=7` is [1, 1, 1, 0].

            decimal : int
                decimal value of a reverse_binary. For instance, the decimal
                value of `reverse_binary=[1, 1, 1, 0]` is 7 (decimal = (
                2**0)*1 + (2**1)*1 + (2**2)*1 + (2**3)*0 = 7).
        """
        x_binary_number: npt.NDArray[np.bool_] = np.zeros(
            self.n_binary_input, bool
        )
        for i in range(self.n_binary_input):
            x_binary_number[i] = decimal % 2
            decimal = decimal // 2
        return x_binary_number

    def get_str_info_function(
        self, prev_feature: int = 1, name_input: str = "feature_"
    ) -> str:
        """
        Return information of `function` in string format.

        Parameters
        ----------
        prev_feature : int
            Index of the first input to the function (just for naming
            purpose).

        name_input : str
            String name to show what is exactly the input to the function.

        Returns
        -------
        str
            Information of `function`.
        """
        print_statement: str = "(["
        for id_feature in range(self.n_binary_input):
            if id_feature != self.n_binary_input - 1:
                print_statement += f"{name_input}{id_feature + prev_feature}, "
            else:
                print_statement += (
                    f"{name_input}"
                    f"{id_feature + prev_feature}]: Binary "
                    f"Output) ->\n"
                )

        for id_state in range(self.n_diff_states):
            binary_input: npt.NDArray[
                np.bool_
            ] = self.__convert_decimal_to_reverse_binary(id_state).astype(int)
            binary_output: Union[int, bool] = int(self.function[id_state])
            if id_state != self.n_diff_states - 1:
                print_statement += f"({binary_input}: {binary_output}), "
            else:
                print_statement += f"({binary_input}: {binary_output})"
        return print_statement

    @staticmethod
    def validate_n_binary_input_binary_function(n_binary_input: Any) -> None:
        """
        Validate `n_binary_input` if it is integer and greater than 0.

        Parameters
        ----------
        n_binary_input : {float, int}
            Input value for `n_binary_input` to be validated.
        """
        if not Base.is_integer(n_binary_input):
            raise TypeError(
                "The input value for n_binary value of BinaryFunction class"
                f" should be 'integer' not {type(n_binary_input)}!"
            )

        if not n_binary_input >= 1:
            raise ValueError(
                "The input value for n_binary value of BinaryFunction class"
                f" should be 'greater than zero', {n_binary_input}<1!"
            )

    @staticmethod
    def validate_binary_input(
        binary_input: Union[list[bool], npt.NDArray[np.bool_]],
        expected_size: int,
    ) -> tuple[bool, npt.NDArray[np.bool_]]:
        """
        Validate `binary_input` as expected to match the `expected_size`.

        Parameters
        ----------
        binary_input : {array-like, ndarray} of bool
            Input array to the binary `function`.
        expected_size : int
            Expected size of the `binary_input` array.

        Returns
        -------
        status : bool
            True if `binary_input` is a valid input.
        binary_input : ndarray of bool
            Validated `binary_input`.
        """
        if not Base.is_array_like(binary_input):
            raise TypeError(
                f"the 'binary_input' should be an array which has len, "
                f"shape or __array__ attribute! Not a {type(binary_input)}"
            )

        try:
            binary_input = np.asarray(binary_input)
        except ValueError:
            raise ValueError(
                "'binary_input' array has an inhomogeneous " "shape!"
            )

        if len(binary_input.shape) == 2:
            if not (
                binary_input.size == expected_size
                and (1 in binary_input.shape)
            ):
                raise ValueError(
                    f"shape of 'binary_input' is not as expected! It is "
                    f"expected to be ({expected_size},1) while it is"
                    f" {binary_input.shape} "
                )
        elif len(binary_input.shape) == 1:
            if not binary_input.size == expected_size:
                raise ValueError(
                    f"shape of 'binary_input' is not as expected! It is "
                    f"expected to be ({expected_size},) while it is"
                    f" {binary_input.shape} "
                )
        else:
            raise ValueError(
                f"dimension of 'binary_input' array should be ("
                f"expected_size,) while the input array has the shape"
                f" {binary_input.shape}"
            )

        if not Base.is_array_binary(binary_input):
            raise ValueError(
                "the 'binary_input' should be a 'binary' array, only 0 "
                "and 1 are acceptable as the elements of the array."
            )

        return True, np.asarray(binary_input, bool).flatten()
