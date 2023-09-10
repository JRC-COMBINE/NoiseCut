"""Generate a structure for the dataset."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>
from typing import Union

import numpy as np
import numpy.typing as npt

from noisecut.tree_structured.base_structured_data import BaseStructuredData
from noisecut.tree_structured.binary_function import BinaryFunction


class StructuredData(BaseStructuredData):
    """
    Generate a structure for the dataset.

    StructuredData class consist of some attributes and methods to keep
    track of the structure of the data with binary functions related to
    black boxes.

    Parameters
    ----------
    n_input_each_box : {list, ndarray} of int of shape (n_box,)
        An array of size `n_box` (number of first-layer black boxes) which
        keeps number of input features to each box.
        For instance, when `n_input_each_box=[2, 4, 1]`, it means there are
        three first-layer black boxes and number of input features to box1,
        box2 and box3 is 2, 4, and 1, respectively.

        Note: There is only one black box in the second-layer.

    Attributes
    ----------
    all_f : list of BinaryFunction of len `n_box`
        List of BinaryFunction objects to store data related to function of
        each first-layer black box. For instance, `all_f[0]` is a variable
        of type BinaryFunction which keeps info of box1 which exists in the
        first-layer black boxes.
    black_box_f : BinaryFunction
        A BinaryFunction object to store info of the second-layer black box.

        Note: There is only one black box in the second-layer.
    """

    def __init__(
        self, n_input_each_box: Union[list[int], npt.NDArray[np.int_]]
    ) -> None:
        super().__init__()

        self.validate_n_input_each_box(n_input_each_box)

        # Initialization
        self.all_f: list[BinaryFunction] = []
        for id_box in range(self.n_box):
            self.all_f.append(BinaryFunction(self.n_input_each_box[id_box]))
        self.black_box_f: BinaryFunction = BinaryFunction(self.n_box)

        self.__i_func: npt.NDArray[np.int_] = np.zeros(self.n_box + 1, int)
        self.__set_start_index_function()

    def __set_start_index_function(self) -> None:
        """
        Set `i_func` array.

        Set `i_func` array which keeps widely-used values in some of the
        methods of this class. For instance, if `n_input_each_box = [2, 4,
        1]`, `i_func` is set to `[0, 0+2, 0+2+4, 0+2+4+1] = [0, 2, 6, 7]`.
        """
        for id_box in range(self.n_box):
            self.__i_func[id_box + 1] = (
                self.n_input_each_box[id_box] + self.__i_func[id_box]
            )

    def set_binary_function_of_box(
        self, id_box: int, func: Union[list[bool], npt.NDArray[np.bool_]]
    ) -> None:
        """
        Return the binary function of the requested box.

        Parameters
        ----------
        id_box : int
            Index of the box, it starts from zero.
        func : ndarray of shape(2**n_input_each_box[id_box],)
            Binary function.
        """
        status, id_box = self.validate_id_box(id_box)
        assert status
        self.all_f[id_box].set_binary_function_manually(func)

    def get_binary_function_of_box(self, id_box: int) -> npt.NDArray[np.bool_]:
        """
        Return the binary function of the requested box.

        Parameters
        ----------
        id_box : int
            Index of the box, it starts from zero.

        Returns
        -------
        ndarray of bool of shape (2**`self.n_input_each_box[id_box]`,)
            Binary function of the `id_box`.
        """
        status, id_box = self.validate_id_box(id_box)
        assert status
        return self.all_f[id_box].function

    def set_binary_function_black_box(
        self, func: Union[list[bool], npt.NDArray[np.bool_]]
    ) -> None:
        """
        Return binary function of the 2nd-layer black box.

        Parameters
        ----------
        func : ndarray of bool of shape (2**n_box,)
            Binary function of the 2nd-layer black box.
        """
        self.black_box_f.set_binary_function_manually(func)

    def get_binary_function_black_box(self) -> npt.NDArray[np.bool_]:
        """
        Return binary function of the 2nd-layer black box.

        Returns
        -------
        ndarray of bool of shape (2**n_box,)
            Binary function of the output box.
        """
        return self.black_box_f.function

    def convert_decimal_index_to_decimal(
        self, decimal_index: npt.NDArray[np.int_]
    ) -> int:
        """
        Convert decimal_index to decimal (see Notes).

        Parameters
        ----------
        decimal_index : ndarray of shape (n_box,)
                See Notes.

        Returns
        -------
        int
            Decimal number.

        Notes
        -----
        To understand the definition of the terms better let's consider,
        `n_box=3` and `n_input_each_box = [2, 4, 1]`. One valid binary
        input feature can be as `reverse_binary = [0, 1, 1, 1, 0, 1, 1]`.

        reverse_binary : ndarray of shape (dimension,)
            Binary representation of a value in which first element in the
            array has the least value.
        decimal_index : ndarray of shape (n_box,)
            Each element of the array stores the decimal representation of
            the binary input feature to each box.
            For instance, based on the above example, `decimal index` of the
            `reverse_binary` is as `[decimal(0, 1), decimal(1, 1, 0, 1),
            decimal(1)] = [2, 11, 1]`.
        decimal : int
            Decimal value of a reverse_binary. For instance, based on the
            above example, the decimal value of `reverse_binary` is `110
            (decimal = (2**0)*0 + (2**1)*1 + (2**2)*1 + (2**3)*1+ (2**4)*0 +
            (2**5)*1 + (2**6)*1 = 110)`.
        """
        number: int = 0
        coefficient: int = 1

        number += decimal_index[0] * coefficient
        for id_box in range(1, self.n_box):
            coefficient *= self.all_f[id_box - 1].n_diff_states
            number += decimal_index[id_box] * coefficient

        return number

    @staticmethod
    def convert_decimal_to_reverse_binary(
        decimal_number: int, dim: int
    ) -> npt.NDArray[np.bool_]:
        """
        Convert decimal to reverse_binary (see Notes).

        Parameters
        ----------
        decimal_number : int
            See Notes.
        dim : int
            Length of the binary number.

        Returns
        -------
        ndarray of bool of shape(dim, )
            Reverse binary.

        Notes
        -----
        To understand the definition of the terms better let's consider,
        `n_box=3` and `n_input_each_box = [2, 4, 1]`. One valid binary
        input feature can be as `reverse_binary = [0, 1, 1, 1, 0, 1, 1]`.

        reverse_binary : ndarray of shape (dimension,)
            Binary representation of a value in which first element in the
            array has the least value.
        decimal_index : ndarray of shape (n_box,)
            Each element of the array stores the decimal representation of
            the binary input feature to each box.
            For instance, based on the above example, `decimal index` of the
            `reverse_binary` is as `[decimal(0, 1), decimal(1, 1, 0, 1),
            decimal(1)] = [2, 11, 1]`.
        decimal : int
            Decimal value of a reverse_binary. For instance, based on the
            above example, the decimal value of `reverse_binary` is `110
            (decimal = (2**0)*0 + (2**1)*1 + (2**2)*1 + (2**3)*1+ (2**4)*0 +
            (2**5)*1 + (2**6)*1 = 110)`.
        """
        x_binary_number: npt.NDArray[np.bool_] = np.zeros(dim, bool)
        for i in range(dim):
            x_binary_number[i] = decimal_number % 2
            decimal_number = decimal_number // 2
        return x_binary_number

    @staticmethod
    def convert_reverse_binary_to_decimal(
        reverse_binary: npt.NDArray[np.bool_],
    ) -> int:
        """
        Convert reverse_binary to decimal(see Notes).

        Parameters
        ----------
        reverse_binary : ndarray of shape (dim,)
            See Notes.

        Returns
        -------
        int
            Decimal value.

        Notes
        -----
        To understand the definition of the terms better let's consider,
        `n_box=3` and `n_input_each_box = [2, 4, 1]`. One valid binary
        input feature can be as `reverse_binary = [0, 1, 1, 1, 0, 1, 1]`.

        reverse_binary : ndarray of shape (dimension,)
            Binary representation of a value in which first element in the
            array has the least value.
        decimal_index : ndarray of shape (n_box,)
            Each element of the array stores the decimal representation of
            the binary input feature to each box.
            For instance, based on the above example, `decimal index` of the
            `reverse_binary` is as `[decimal(0, 1), decimal(1, 1, 0, 1),
            decimal(1)] = [2, 11, 1]`.
        decimal : int
            Decimal value of a reverse_binary. For instance, based on the
            above example, the decimal value of `reverse_binary` is `110
            (decimal = (2**0)*0 + (2**1)*1 + (2**2)*1 + (2**3)*1+ (2**4)*0 +
            (2**5)*1 + (2**6)*1 = 110)`.
        """
        length: int = len(reverse_binary)

        decimal_number: int = 0
        for i in range(length):
            decimal_number += 2**i * reverse_binary[i]
        return decimal_number

    def calc_output_structured_system(
        self, x_binary_number: npt.NDArray[np.bool_]
    ) -> np.bool_:
        """
        Calculate output of the structured system for `x_binary_number`.

        Parameters
        ----------
        x_binary_number : ndarray of bool of shape (dimension,)
            Input binary array to the system.

        Returns
        -------
        bool
            Binary output for `x_binary_number`.
        """
        input_black_box: npt.NDArray[np.bool_] = np.zeros(self.n_box, bool)

        for id_box in range(self.n_box):
            binary_input_to_id_box_func = x_binary_number[
                self.__i_func[id_box] : self.__i_func[id_box + 1]
            ]
            input_black_box[id_box] = self.all_f[id_box].calc_output_func(
                binary_input_to_id_box_func
            )
        return self.black_box_f.calc_output_func(input_black_box)

    def calc_decimal_input_structured_system_to_black_box(
        self, x_binary_number: npt.NDArray[np.bool_]
    ) -> int:
        """
        Return decimal value of the binary input to the second-layer black box.

        Parameters
        ----------
        x_binary_number : ndarray of shape (dimension,)
            Array of binary values as the input of the structured system.

        Returns
        -------
        int
            Decimal value.

        Examples
        --------
        Let's consider, `n_box=3` and `n_input_each_box = [2, 4, 1]`. One
        valid binary input feature can be as `reverse_binary = [0, 1, 1, 1,
        0, 1, 1]`. decimal value to the 2nd-layer black box is calculated
        as follows: `decimal( [func_box1(0, 1), func_box1(1, 1, 0, 1),
        func_box3(1)] )=decimal( [0, 1, 1] ) = 6`.
        """
        input_black_box: npt.NDArray[np.bool_] = np.zeros(self.n_box, bool)

        for id_box in range(self.n_box):
            binary_input_to_id_box_func = x_binary_number[
                self.__i_func[id_box] : self.__i_func[id_box + 1]
            ]
            input_black_box[id_box] = self.all_f[id_box].calc_output_func(
                binary_input_to_id_box_func
            )
        return self.convert_reverse_binary_to_decimal(input_black_box)

    def calc_output_structured_system_multiple_input(
        self, multiple_x_binary: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """
        Return output of the structured system to more than one input.

        Parameters
        ----------
        multiple_x_binary : ndarray of shape (dimension,n_samples)
            Each row of the `multiple_x_binary` array is a binary input
            value to the structured system.

        Returns
        -------
        ndarray of shape (n_samples,)
            Output of the system in one_to_one mapping of binary input
            `multiple_x_binary`.
        """
        n_samples: int = len(multiple_x_binary)
        y_number: npt.NDArray[np.bool_] = np.zeros(n_samples, bool)

        for id_sample in range(n_samples):
            y_number[id_sample] = self.calc_output_structured_system(
                multiple_x_binary[id_sample][:]
            )
        return y_number

    def build_complete_data_set_y_array(self) -> npt.NDArray[np.bool_]:
        """
        Return complete dataset output.

        Return output of the structured system to any combination of input
        features in an order. Index of the output array is equivalent to
        decimal value of the reverse binary input.

        Returns
        -------
        ndarray of bool of shape (2**dimension,)
            Output of the system.

        Examples
        --------
        For instance, consider a structured system with `dimension=5`. `y[
        1]` is the output of the structured system to binary input `[1, 0,
        0, 0, 0]`. `y[22]` is the output of the structured system to binary
        input `[1, 0, 1, 1, 0]`.
        """
        y_number: npt.NDArray[np.bool_] = np.zeros(2**self.dimension, bool)

        for number in range(2**self.dimension):
            x_binary_number = self.convert_decimal_to_reverse_binary(
                number, self.dimension
            )
            y_number[number] = self.calc_output_structured_system(
                x_binary_number
            )

        return y_number

    def get_complete_data_set(
        self, file_name: Union[str, None] = None
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """
        Return and write output of the system to all possible input binaries.

        Parameters
        ----------
        file_name : str, default=None
            Name of the file (with or without the path) in which you aim to
            store results. If `file_name=None`, dataset will not be written
            in a file.

        Returns
        -------
        x : ndarray of bool of shape (2**dimension, dimension)
            Binary input dataset.
        y : ndarray of bool of shape (2**dimension,)
            Binary output dataset in one_to_one mapping of `x`.
        """
        x: npt.NDArray[np.bool_] = np.zeros(
            (2**self.dimension, self.dimension), bool
        )
        y_number: npt.NDArray[np.bool_] = np.zeros(2**self.dimension, bool)

        for number in range(2**self.dimension):
            x[number][:] = self.convert_decimal_to_reverse_binary(
                number, self.dimension
            )
            y_number[number] = self.calc_output_structured_system(x[number][:])

        if file_name is not None:
            with open(file_name, "w") as file:
                for id_box in range(self.n_box):
                    file.write(str(self.n_input_each_box[id_box]) + "    ")
                file.write("\n")
                for number in range(2**self.dimension):
                    for d in range(self.dimension):
                        file.write(str(int(x[number][d])) + "    ")
                    file.write(str(int(y_number[number])) + "\n")

        return x, y_number

    def print_binary_function_model(self) -> None:
        """Print all functions of the structured model."""
        prev_feature: int = 1
        for id_box in range(self.n_box):
            print(f"Function Box{id_box+1}")
            print(self.all_f[id_box].get_str_info_function(prev_feature))
            prev_feature += self.n_input_each_box[id_box]

        print("Function Black Box")
        print(self.black_box_f.get_str_info_function(name_input="Output_box_"))
