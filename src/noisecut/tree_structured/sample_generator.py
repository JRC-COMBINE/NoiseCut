"""Sample Generator class."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>
from typing import Union

import numpy as np
import numpy.typing as npt

from noisecut.tree_structured.structured_data import StructuredData


class SampleGenerator(StructuredData):
    """
    A class to generate synthetic data.

    SampleGenerator is used for building synthetic structured binary data. it
    can be done randomly by setting the structure of data or manually by
    setting the function of each box.

    Parameters
    ----------
    n_input_each_box : {list, ndarray} of shape (n_box,)
        An array of size `n_box` (number of first-layer black boxes) which
        keeps number of input features to each box.
        For instance, when `n_input_each_box=[2, 4, 1]`, it means there are
        three first-layer black boxes and number of input features to box1,
        box2 and box3 is 2, 4, and 1, respectively.

        Note: There is only one black box in the second-layer.
    allowance_rand : bool, default=False
        If True, all functions of the binary structure are set randomly.

    Attributes
    ----------
    functionality : bool
        Structured binary data has functionality if it is set to True.
    decimal_index_boxes : ndarray of shape (n_box,)
        Keeps the decimal value of input features to each first-layer black
        box. For instance, if `n_input_each_box = [2, 4, 1]` and one sample
        has the following input feature [0, 1, 0, 0, 1, 1, 1],
        the `decimal_index_boxes` is as follows: `[decimal(0, 1), decimal(0,
        0, 1, 1), decimal(1)] = [2,12,1]`
    """

    def __init__(
        self,
        n_input_each_box: Union[list[int], npt.NDArray[np.int_]],
        allowance_rand: bool = False,
    ) -> None:
        super().__init__(n_input_each_box)

        self.functionality: bool = False
        self.decimal_index_boxes: npt.NDArray[np.int_] = np.zeros(
            self.n_box, int
        )

        if allowance_rand:
            self.set_rand_func()

    def set_rand_func(self) -> None:
        """Set function to all black boxes randomly."""
        self.functionality = False
        while not self.functionality:
            for id_box in range(self.n_box):
                self.all_f[id_box].set_random_binary_function()
            self.black_box_f.set_random_binary_function()

            self.functionality = self.has_synthetic_example_functionality()

    def has_synthetic_example_functionality(self) -> bool:
        """
        Check the functionality of the tree-structured binary dataset.

        Returns
        -------
        bool
            True if the dataset has functionality.
        """
        self.y: npt.NDArray[np.bool_] = self.build_complete_data_set_y_array()
        count_true: int = 0
        for id_box in range(self.n_box):
            self.functionality = False
            self.__check_functionality_in_each_box_by_recursion(
                (id_box + 1) % self.n_box
            )
            count_true += self.functionality

        return count_true == self.n_box

    def __check_functionality_in_each_box_by_recursion(
        self, i_o: int, number_loop_level: int = 0
    ) -> None:
        """
        Check whether the set synthetic data has functionality for a black box.

        Parameters
        ----------
        i_o : int
            Indicates the sequence of the nested loops. For instance,
            when `n_box=3`, if `i_o=0`, the sequence of nested loop is like:

            for i[0] in range(...):
                for i[1] in range(...):
                    for i[2] in range(...):
                        ...

            If `i_o=1`, the sequence of nested loop is like:

            for i[1] in range(...):
                for i[2] in range(...):
                    for i[0] in range(...):
                        ...

           And if `i_o=2`, the sequence of nested loop is like:

            for i[2] in range(...):
                for i[0] in range(...):
                    for i[1] in range(...):
                        ...

            Hint: (i[0], i[1], i[2]) can be seen as (i, j, k).
        number_loop_level : int, default=0
            Number of loop in which code runs. Purpose of this function
            is to build nested loops to the number of `n_box`. For
            instance, when `n_box=3` and `i_o=0`, it is somehow similar
            to building such a for loop:

            for i[0] in range(...): -> number_loop_level = 0
                for i[1] in range(...): -> number_loop_level = 1
                    for i[2] in range(...): -> number_loop_level = 2
                        ...
            hint: (i[0], i[1], i[2]) can be seen as (i, j, k).
        """
        id_box: int = (number_loop_level + i_o) % self.n_box

        if number_loop_level == self.n_box - 1:
            self.decimal_index_boxes[id_box] = 0
            decimal: int = self.convert_decimal_index_to_decimal(
                self.decimal_index_boxes
            )
            temp: bool = self.y[decimal]

            for self.decimal_index_boxes[id_box] in range(  # noqa: B007, B020
                1, self.all_f[id_box].n_diff_states
            ):
                decimal = self.convert_decimal_index_to_decimal(
                    self.decimal_index_boxes
                )
                if self.y[decimal] != temp:
                    self.functionality = True
                    break

        else:
            for self.decimal_index_boxes[id_box] in range(  # noqa: B007, B020
                self.all_f[id_box].n_diff_states
            ):
                if self.functionality:
                    break
                self.__check_functionality_in_each_box_by_recursion(
                    i_o, number_loop_level + 1
                )
