"""
Coder class for implementing NoiseCut method.

ATTENTION :
    It is highly recommended to use NoiseCut class in noisecut_model.py to
    construct the model as the access of the user has been limited to some
    inherited methods and attributes of the class to keep intact attributes
    from unintended manipulations.
"""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>
from typing import Union

import numpy as np
import numpy.typing as npt

from noisecut.max_cut.solvers import MaxCutSolvers
from noisecut.tree_structured.base import Base
from noisecut.tree_structured.base_structured_data import BasePseudoBooleanFunc
from noisecut.tree_structured.structured_data import StructuredData


class CoderNoiseCut(StructuredData):
    """
    NoiseCut model to fit the tree-structured binary dataset.

    CoderNoiseCut class is used to predict the binary functions of the
    structured model of the training dataset by NoiseCut method. However,
    it is highly recommended to use NoiseCut class in noisecut_model.py file
    to construct the model as the access of the user has been limited to
    some inherited methods and attributes of the class to keep intact the
    attributes from unintended manipulations.

    Parameters
    ----------
    n_input_each_box : {list, ndarray} of shape (n_box,)
        An array of size `n_box` (number of first-layer black boxes) which
        keeps number of input features to each box. For instance,
        when `n_input_each_box=[2, 4, 1]`, it means there are three
        first-layer black boxes and number of input features to box1,
        box2 and box3 is 2, 4, and 1, respectively.
    threshold : float in range (0,1), default=0.5≤
        Calculation of the function of the 2nd layer black box is done
        based on the `threshold` value. It is more reasonable to be set
        to 0.5 (default value).

    Attributes
    ----------
    y_number_of_0 : ndarray of int of shape (2**dimension,)
        Based on the fact that the NoiseCut method is designed for binary
        data, only 0 and 1 are the valid inputs of the model. To store data
        efficiently, each binary input can be seen as a decimal value which
        is used as the index of the `y_number_of_0` and `y_number_of_1`. If
        the output of the binary input is 0, `y_number_of_0[decimal index]`
        is increased by 1 at that specific decimal index.
    y_number_of_1 : ndarray of int of shape (2**dimension,)
        Based on the fact that the NoiseCut method is designed for binary
        data, only 0 and 1 are the valid inputs of the model. To store data
        efficiently, each binary input can be seen as a decimal value which
        is used as the index of the `y_number_of_0` and `y_number_of_1`. If
        the output of the binary input is 1, `y_number_of_1[decimal index]`
        is increased by 1 at that specific decimal index.
    model_fitted : bool
        True if model is fitted.
    number_of_0_labels : ndarray of int of shape (2**n_box)
        It is used in counting number of 0 of target output values based on
        the binary input to the 2nd-layer black box. Indeed, outputs of the
        1st-layer black boxes enters as an input to the 2nd-layer black box.
        Each binary input to the 2nd-layer black box can be seen as a
        decimal value for indexing `number_of_0_labels`.
    number_of_1_labels : ndarray of int of shape (2**n_box)
        It is used in counting number of one of target output values based on
        the binary input to the 2nd-layer black box. Indeed, outputs of the
        1st-layer black boxes enters as an input to the 2nd-layer black box.
        Each binary input to the 2nd-layer black box can be seen as a
        decimal value for indexing `number_of_1_labels`.
    decimal_index_boxes : ndarray of int of shape (n_box,)
        Keeps the decimal value of input features to each first-layer black
        box. For instance, if `n_input_each_box = [2, 4, 1]` and one sample
        has the following input feature [0, 1, 0, 0, 1, 1, 1],
        the `decimal_index_boxes` is as follows: [decimal(0, 1), decimal(0,
        0, 1, 1), decimal(1)] = [2,12,1].
    probability_of_being_1 : ndarray of shape (2**n_box)
        Is used for setting probability of the target output to be one based
        on the `number_of_0_labels` and `number_of_1_labels`.
    score_of_being_1 : ndarray of shape (2**n_box)
        Is used for setting a score in a defined range (for instance from 1
        to 5) to the target output of a binary input based on the binary
        input to the 2nd layer black box. The score value indicates
        probability of being one. For instance, if the score value is
        maximum in a defined range, it is more likely that the NoiseCut
        model has a good prediction of the output.

     Notes
     -----
     There is only one black box in the second-layer.
    """

    def __init__(
        self,
        n_input_each_box: Union[list[int], npt.NDArray[np.int_]],
        threshold: float = 0.5,
    ) -> None:
        super().__init__(n_input_each_box)

        _, self.threshold = self.validate_threshold(threshold)

        self.y_number_of_0: npt.NDArray[np.int_] = np.zeros(
            2**self.dimension, int
        )
        self.y_number_of_1: npt.NDArray[np.int_] = np.zeros(
            2**self.dimension, int
        )

        self.model_fitted: bool = False
        self.w_initialized: bool = False
        self.data_exist: bool = False
        self.probability_known: bool = False

        self.number_of_0_labels: npt.NDArray[np.int_] = np.zeros(
            2**self.n_box, int
        )
        self.number_of_1_labels: npt.NDArray[np.int_] = np.zeros(
            2**self.n_box, int
        )

        self.decimal_index_boxes: npt.NDArray[np.int_] = np.zeros(
            self.n_box, int
        )

        self.probability_of_being_1: npt.NDArray[np.float_] = np.zeros(
            2**self.n_box, float
        )
        self.score_of_being_1: npt.NDArray[np.int_] = np.zeros(
            2**self.n_box, int
        )

    def initialize_w(self) -> None:
        """Initialize `w`."""
        if not self.w_initialized:
            for id_box in range(self.n_box):
                dim = self.all_f[id_box].n_diff_states
                self.all_f[id_box].w = np.zeros(dim * (dim - 1) // 2, int)
            self.w_initialized = True
        else:
            for id_box in range(self.n_box):
                self.all_f[id_box].w[:] = 0

    def set_weight_each_box_by_recursion(
        self, i_o: int, number_loop_level: int = 0
    ) -> None:
        """
        Set weight of each 1st-layer black box by recursion (NoiseCut method).

        Parameters
        ----------
        i_o : int
            Indicates the sequence of the nested loops. For instance,
            when `n_box=3`, if `i_o=0`, the sequence of nested loop is like:

            for i[0] in range(...):
                for i[1] in range(...):
                    for i[2] in range(...):
                        ...

            and the method sets weight for the 3rd box because of the last
            loop in the nested loop.

            If `i_o=1`, the sequence of nested loop is like:

            for i[1] in range(...):
                for i[2] in range(...):
                    for i[0] in range(...):
                        ...

            and sets weight for the 1st box.

            And if `i_o=2`, the sequence of nested loop is like:

            for i[2] in range(...):
                for i[0] in range(...):
                    for i[1] in range(...):
                        ...

            and sets weight for the 2nd box.

            Hint: (i[0], i[1], i[2]) can be seen as (i, j, k).
        number_loop_level : int, default=0
            Number of loop in which code runs. For instance, purpose of
            this function is to build nested loops to the number of
            `n_box`. For instance, when `n_box=3` and `i_o=0`, it is
            somehow similar to building such a for loop:

            for i[0] in range(...): -> number_loop_level = 0
                for i[1] in range(...): -> number_loop_level = 1
                    for i[2] in range(...): -> number_loop_level = 2
                        ...
            hint: (i[0], i[1], i[2]) can be seen as (i, j, k).
        """
        id_box: int = (number_loop_level + i_o) % self.n_box

        if number_loop_level == self.n_box - 1:
            n_s: int = self.all_f[id_box].n_diff_states
            number_iter: int = 0

            index_w: int = 0
            for self.decimal_index_boxes[id_box] in range(n_s - 1):
                decimal_m: int = self.convert_decimal_index_to_decimal(
                    self.decimal_index_boxes
                )

                decimal_index_boxes_input_n: npt.NDArray[
                    np.int_
                ] = self.decimal_index_boxes.copy()

                for j in range(self.decimal_index_boxes[id_box] + 1, n_s):
                    decimal_index_boxes_input_n[id_box] = j

                    decimal_n: int = self.convert_decimal_index_to_decimal(
                        decimal_index_boxes_input_n
                    )

                    self.all_f[id_box].w[index_w] += (
                        self.y_number_of_1[decimal_n]
                        * self.y_number_of_0[decimal_m]
                        + self.y_number_of_0[decimal_n]
                        * self.y_number_of_1[decimal_m]
                    )

                    number_iter += 1
                    index_w += 1
        else:
            for self.decimal_index_boxes[id_box] in range(  # noqa: B007, B020
                self.all_f[id_box].n_diff_states
            ):
                self.set_weight_each_box_by_recursion(
                    i_o, number_loop_level + 1
                )

    def count_labels_by_recursion(self, number_loop_level: int = 0) -> None:
        """
        Count labels for each binary input to the 2nd-layer black box.

        Count number of 0 and 1 of target output values based on the binary
        input to the 2nd-layer black box. `number_of_0_labels` and
        `number_of_1_labels` are set in this method.

        Parameters
        ----------
        number_loop_level : int, default=0
            Number of loop in which code runs. Purpose of this function
            is to build nested loops to the number of `n_box`.
            For instance, when `n_box=3`, it is somehow similar to building
            such a for loop:

            for i[0] in range(...): -> number_loop_level = 0
                for i[1] in range(...): -> number_loop_level = 1
                    for i[2] in range(...): -> number_loop_level = 2
                        ...
            hint: (i[0], i[1], i[2]) can be seen as (i, j, k).
        """
        id_box: int = number_loop_level % self.n_box

        if number_loop_level == self.n_box - 1:
            n_s: int = self.all_f[id_box].n_diff_states

            for self.decimal_index_boxes[id_box] in range(n_s):  # noqa: B007
                decimal_m: int = self.convert_decimal_index_to_decimal(
                    self.decimal_index_boxes
                )
                if (
                    self.y_number_of_0[decimal_m]
                    or self.y_number_of_1[decimal_m]
                ):
                    decimal_number_black_box: int = 0
                    for j in range(self.n_box):
                        decimal_number_black_box += (
                            2**j
                            * self.all_f[j].function[
                                self.decimal_index_boxes[j]
                            ]
                        )

                    self.number_of_0_labels[
                        decimal_number_black_box
                    ] += self.y_number_of_0[decimal_m]
                    self.number_of_1_labels[
                        decimal_number_black_box
                    ] += self.y_number_of_1[decimal_m]

        else:
            for self.decimal_index_boxes[id_box] in range(  # noqa: B007, B020
                self.all_f[id_box].n_diff_states
            ):
                self.count_labels_by_recursion(number_loop_level + 1)

    def coder_set_training_data(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
    ) -> None:
        """
        Set `x` and `y` dataset.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, dimension)
            Each row of the array `x` is a binary input.
        y : {array-like, ndarray, dataframe} of shape (n_data,)
            Target output value in one_to_one mapping of binary input `x`.

        Warns
        -----
        If some data has already been set, by calling
        `coder_set_training_data()` again, the previous data will be
        deleted. If you do not intend to delete previous data,
        use `coder_add_training_data()` instead.
        """
        if self.data_exist:
            raise Warning(
                "Some data has been set before! Now by calling "
                "set_training_data, the previous input data has been "
                "deleted! Note: If you do not intend to delete previous "
                "data, use add_training_data() instead of set_training_data("
                ") and run the model again."
            )
            self.y_number_of_0[:] = 0
            self.y_number_of_1[:] = 0
            self.model_fitted = False
            self.probability_known = False

        _, x_validated, y_validated = self.validate_x_y(x, y)

        n_samples = len(y_validated)

        if n_samples == 1:
            x_validated = x_validated.flatten()
            decimal_number: int = self.convert_reverse_binary_to_decimal(
                x_validated
            )
            if y_validated[0]:
                self.y_number_of_1[decimal_number] += 1
            else:
                self.y_number_of_0[decimal_number] += 1
        else:
            for i in range(n_samples):
                decimal_number = self.convert_reverse_binary_to_decimal(
                    x_validated[i]
                )
                if y[i]:
                    self.y_number_of_1[decimal_number] += 1
                else:
                    self.y_number_of_0[decimal_number] += 1
        self.data_exist = True

    def coder_add_training_data(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
    ) -> None:
        """
        Add `x` and `y` dataset to the existing dataset.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, dimension)
            Each row of the array `x` is a binary input.
        y : {array-like, ndarray, dataframe} of shape (n_data,)
            Target output value in one_to_one mapping of binary input `x`.

        Raises
        ------
        TypeError
            If some data has already been set, it will work. Otherwise,
            it pops up an error.
        """
        if not self.data_exist:
            raise TypeError("No Data exist to add to! First set data!")

        _, x_validated, y_validated = self.validate_x_y(x, y)
        self.model_fitted = False
        self.probability_known = False

        n_samples = len(y_validated)
        decimal_number: int

        if n_samples == 1:
            x_validated = x_validated.flatten()
            decimal_number = self.convert_reverse_binary_to_decimal(
                x_validated
            )
            if y_validated[0]:
                self.y_number_of_1[decimal_number] += 1
            else:
                self.y_number_of_0[decimal_number] += 1
        else:
            for i in range(n_samples):
                decimal_number = self.convert_reverse_binary_to_decimal(
                    x_validated[i]
                )
                if y[i]:
                    self.y_number_of_1[decimal_number] += 1
                else:
                    self.y_number_of_0[decimal_number] += 1

    def coder_fit(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
        with_more_data: bool = False,
        print_result: bool = False,
        print_weights: bool = False,
    ) -> None:
        """
        Fit the model based on the input dataset.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, dimension)
            Training data to fit the NoiseCut model. Each row of the array
            `x` is a binary input.
        y : {array-like, ndarray, dataframe} of shape (n_data,)
            Training data output in one_to_one mapping of binary training
            data input `x`.
        with_more_data : bool, default=False
            Whether to fit the model with additional data if the model has
            been fitted once. Indeed, the `x`, `y` will be added to the
            existing data from the previous fitting and then the model will
            fit again if `with_more_data=True`.
        print_result : bool, default=False
            Whether to print the result of fitting.
        print_weights : bool, default=False
            Whether to print the set weights of the 1st-layer black boxes
            (for debugging purpose).
        """
        if with_more_data:
            self.coder_add_training_data(x, y)
            self.number_of_0_labels[:] = 0
            self.number_of_1_labels[:] = 0
        else:
            self.coder_set_training_data(x, y)
        self.initialize_w()
        maxcut_solver = MaxCutSolvers()

        for id_box in range(self.n_box):
            if self.n_input_each_box[id_box] == 1:
                self.set_binary_function_of_box(id_box, [0, 1])
            else:
                self.set_weight_each_box_by_recursion(
                    (id_box + 1) % self.n_box
                )
                if print_weights:
                    print("")
                    print(f"Weight of Box{id_box+1}")
                    temp = 0
                    for i in range(self.all_f[id_box].n_diff_states):
                        for j in range(
                            i + 1, self.all_f[id_box].n_diff_states
                        ):
                            print(
                                f"{i+1}, {j+1}, "
                                f"{self.all_f[id_box].w[temp]}"
                            )
                            temp += 1
                maxcut_solver.set_weight_1d_and_n_vertices(
                    weight=self.all_f[id_box].w,
                    n_vertices=self.all_f[id_box].n_diff_states,
                )

                self.set_binary_function_of_box(
                    id_box, maxcut_solver.solve_maxcut()[1]
                )

        self.count_labels_by_recursion()

        b_b_func: npt.NDArray[np.bool_] = np.zeros(2**self.n_box, bool)
        for i in range(2**self.n_box):
            if (
                self.number_of_1_labels[i] == 0
                and self.number_of_0_labels[i] == 0
            ):
                b_b_func[i] = True
            else:
                b_b_func[i] = (
                    self.number_of_1_labels[i]
                    / (self.number_of_1_labels[i] + self.number_of_0_labels[i])
                ) >= self.threshold

        self.set_binary_function_black_box(b_b_func)
        if print_result:
            self.print_binary_function_model()
        self.model_fitted = True

    def coder_predict(
        self, x: Union[list[bool], npt.NDArray[np.bool_]]
    ) -> npt.NDArray[np.bool_]:
        """
        Return predicted output of the NoiseCut model to the binary inout `x`.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_test_data, dimension)
            Data input to predict the binary output for each row of it, which
            is a binary input.

        Returns
        -------
        ndarray of bool of shape (n_test_data,)
            Predicted output in one_to_one mapping of binary input `x`.
        """
        if not self.model_fitted:
            raise RuntimeError(
                "Model has not been fitted yet! "
                "call predict(x) after fitting model."
            )
        status, x = self.validate_x(x)
        assert status
        return self.calc_output_structured_system_multiple_input(x)

    def coder_predict_all(self) -> npt.NDArray[np.bool_]:
        """
        Return all predicted output of the structured system by NoiseCut.

        Predict output of the structured system for any possible
        binary input based on the fitted NoiseCut model. Decimal value of
        binary input can be seen as an index for the returned array to get
        the predicted value for that binary input.

        For instance, consider a structured system with 'dimension=5':

        `returned_array[ 1]` is the predicted output of the fitted NoiseCut
        model to binary input `[1, 0, 0, 0, 0]`.

        `returned_array[22]` is the predicted output of the fitted NoiseCut
        model to binary input `[1, 0, 1, 1, 0]`.

        Returns
        -------
        ndarray of shape (2**dimension,)
            Predicted Output.
        """
        if not self.model_fitted:
            raise RuntimeError(
                "Model has not been fitted yet! call "
                "predict_all(x) after fitting model."
            )

        return self.build_complete_data_set_y_array()

    def coder_set_uncertainty_measure(
        self, file_path_result: Union[str, None] = None
    ) -> None:
        """
        Set uncertainty.

        Parameters
        ----------
        file_path_result : str, default=None
            Path of a file to save the uncertainty result to it.
        """
        self.__set_probability_of_being_1()
        print("Binary input black box, number of 0, number of 1")
        for i in range(2**self.n_box):
            print(
                "{}, {}, {}, {}".format(
                    np.asarray(
                        self.convert_decimal_to_reverse_binary(i, self.n_box),
                        int,
                    ),
                    self.number_of_0_labels[i],
                    self.number_of_1_labels[i],
                    self.probability_of_being_1[i],
                )
            )

        if file_path_result is not None:
            with open(file_path_result, "w") as file:
                for id_box in range(self.n_box):
                    file.write(f"Box_{id_box}, ")

                file.write("number of 0, number of 1, Probability 1\n")
                for i in range(2**self.n_box):
                    r_b = self.convert_decimal_to_reverse_binary(i, self.n_box)
                    for id_box in range(self.n_box):
                        file.write(f"{int(r_b[id_box])}, ")
                    file.write(
                        f"{self.number_of_0_labels[i]}, "
                        f"{self.number_of_1_labels[i]}, "
                        f"{self.probability_of_being_1[i]}\n"
                    )

    def __set_probability_of_being_1(self) -> None:
        """
        Set probability of the target output to be 1.

        Set probability of the target output to be 1 based on the binary
        input to the 2nd-layer black box, which is the output of the
        1st-layer black boxes. `number_of_0_labels` and `number_of_1_labels`
        are used to set this measure for each possible binary input to the
        2nd-layer black box.
        """
        if not self.model_fitted:
            raise RuntimeError(
                "Model has not been fitted yet! call "
                "predict_all(x) after fitting model."
            )

        if not self.probability_known:
            for i in range(2**self.n_box):
                if (
                    self.number_of_1_labels[i] == 0
                    and self.number_of_0_labels[i] == 0
                ):
                    self.probability_of_being_1[i] = 0.5
                else:
                    self.probability_of_being_1[i] = float(
                        self.number_of_1_labels[i]
                    ) / float(
                        self.number_of_0_labels[i] + self.number_of_1_labels[i]
                    )
            self.probability_known = True

    def coder_predict_probability_of_being_1(
        self, x: Union[list[bool], npt.NDArray[np.bool_]]
    ) -> npt.NDArray[np.float_]:
        """
        Return probability of the predicted target output to be 1.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_samples, dimension)
            Data input to predict probability of being 1 for each row of it,
            which is a binary input.

        Returns
        -------
        ndarray of float of shape (n_samples,)
            Probability of the predicted target output to be 1 in one_to_one
            mapping of binary input `x`.
        """
        _, x_validated = self.validate_x(x)
        self.__set_probability_of_being_1()

        if (len(x_validated.shape) == 1) or (1 in x_validated.shape):
            n_samples = 1
        else:
            n_samples = len(x_validated)

        p1 = np.zeros(n_samples)

        if n_samples == 1:
            x_validated = x_validated.flatten()
            p1 = self.probability_of_being_1[
                self.calc_decimal_input_structured_system_to_black_box(
                    x_validated
                )
            ]
        else:
            for i in range(n_samples):
                p1[i] = self.probability_of_being_1[
                    self.calc_decimal_input_structured_system_to_black_box(
                        x_validated[i]
                    )
                ]

        return p1

    def coder_predict_pseudo_boolean_func_coef(self) -> npt.NDArray[np.float_]:
        """
        Return the Pseudo boolean coefficients of 2nd-layer black box.

        Return the Pseudo boolean coefficients of the Pseudo boolean
        function of the 2nd-layer black box based on the values of the
        `probability_of_being_1`.

        Returns
        -------
        ndarray of float of shape (2**n_box,)
            Pseudo boolean coefficients in array type.
        """
        self.__set_probability_of_being_1()

        x: npt.NDArray[np.bool_] = np.zeros(
            (2**self.n_box, self.n_box), bool
        )
        for i in range(2**self.n_box):
            x[i] = self.convert_decimal_to_reverse_binary(i, self.n_box)

        pseudo_boolean_func = PseudoBooleanFunc(self.n_box)
        return pseudo_boolean_func.get_coef_boolean_func(
            x, self.probability_of_being_1
        )

    def __set_score(
        self, vector_n_score: Union[list[float], npt.NDArray[np.float_]]
    ) -> None:
        """
        Set score based on probability value.

        Score is set for each binary input of the 2nd-layer black box
        based on the probability value.

        Parameters
        ----------
        vector_n_score : {array-like, ndarray}
            An indicator how the score is given to a range of probabilities.
            `vector_n_score` array Should have 0 as the first element of
            the array and 1 as the last element. The other elements in
            between should be unique and in range 0 to 1. The array should
            be also sorted from lowest, which is zero, to highest, which
            is one.
            For instance, consider `vector_n_score` is `[0, 0.2, 0.4, 0.6,
            0.8, 1]`. Highest score is set to binary inputs of the 2nd-layer
            black box, which has a probability in range 0.8 to 1. Lowest
            score is always one and highest score depends on the length of
            `vector_n_score`. In this example, highest score is 5.
        """
        n_score: int = len(vector_n_score) - 1
        self.__set_probability_of_being_1()
        for i in range(2**self.n_box):
            if self.probability_of_being_1[i] == 1:
                self.score_of_being_1[i] = n_score
            else:
                for j in range(n_score):
                    if (
                        vector_n_score[j]
                        <= self.probability_of_being_1[i]
                        < vector_n_score[j + 1]
                    ):
                        self.score_of_being_1[i] = j + 1

    def coder_predict_score(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        vector_n_score: Union[list[float], npt.NDArray[np.float_]],
        validate: bool = True,
    ) -> npt.NDArray[np.int_]:
        """
        Return predicted score for each binary input of `x`.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_samples, dimension)
            Input data to predict score of output for it. Each row of
            the array `x` is a binary input.
        vector_n_score : {array-like, ndarray}
            An indicator how the score is given to a range of probabilities.
            `vector_n_score` array Should have 0 as the first element of
            the array and 1 as the last element. The other elements in
            between should be unique and in range 0 to 1. The array should
            be also sorted from lowest, which is zero, to highest, which
            is one.
            For instance, consider `vector_n_score` is `[0, 0.2, 0.4, 0.6,
            0.8, 1]`. Highest score is set to binary inputs of the 2nd-layer
            black box, which has a probability in range 0.8 to 1. Lowest
            score is always one and highest score depends on the length of
            `vector_n_score`. In this example, highest score is 5.
        validate : bool
            Whether input data needs to be validated.
            Note: Always validate data if it has not been validated before.

        Returns
        -------
        ndarray of int of shape (n_samples,)
            Score array is set in one_to_one mapping of binary input `x`.
        """
        if validate:
            _, x_validated = self.validate_x(x)
        else:
            x_validated = x
        status, vector_n_score = self.validate_vector_n_score(vector_n_score)
        assert status

        self.__set_score(vector_n_score)

        n_samples: int
        if len(x_validated.shape) == 1 or (1 in x_validated.shape):
            n_samples = 1
        else:
            n_samples = len(x_validated)

        score_: npt.NDArray[np.int_] = np.zeros(n_samples, int)

        if n_samples == 1:
            x_validated = x_validated.flatten()
            score_ = self.score_of_being_1[
                self.calc_decimal_input_structured_system_to_black_box(
                    x_validated
                )
            ]
        else:
            for i in range(n_samples):
                score_[i] = self.score_of_being_1[
                    self.calc_decimal_input_structured_system_to_black_box(
                        x_validated[i]
                    )
                ]

        return score_

    def coder_predict_mortality_of_each_score(
        self,
        x_test: Union[list[bool], npt.NDArray[np.bool_]],
        y_test: Union[list[bool], npt.NDArray[np.bool_]],
        vector_n_score: Union[list[float], npt.NDArray[np.float_]],
        print_mortality: bool = False,
    ) -> tuple[
        npt.NDArray[np.float_], npt.NDArray[np.int_], npt.NDArray[np.int_]
    ]:
        """
        Return percentage of output to be 1 for test data with specific score.

        Parameters
        ----------
        x_test : {array-like, ndarray, dataframe} of shape (n_data,
            dimension)
            Test data input against the fitted NoiseCut model. Each row of
            the array `x` is a binary input.
        y_test : {array-like, ndarray, dataframe} of shape (n_data,)
            Predicted output of test data in one_to_one mapping of test
            data `x`.
        vector_n_score : {array-like, ndarray}
            An indicator how the score is given to a range of probabilities.
            `vector_n_score` array Should have 0 as the first element of
            the array and 1 as the last element. The other elements in
            between should be unique and in range 0 to 1. The array should
            be also sorted from lowest, which is zero, to highest, which
            is one.
            For instance, consider `vector_n_score` is `[0, 0.2, 0.4, 0.6,
            0.8, 1]`. Highest score is set to binary inputs of the 2nd-layer
            black box, which has a probability in range 0.8 to 1. Lowest
            score is always one and highest score depends on the length of
            `vector_n_score`. In this example, highest score is 5.
        print_mortality : bool, default=False
            Whether print result of the method.

        Returns
        -------
        ndarray of shape (n_score,)
            Percentage for each score. For example: `returned_array[0]`
            -> for score 1.

        Raises
        ------
        RuntimeError
            Riase error if model is not fitted.
        """
        _, x_test, y_test = self.validate_x_y(x_test, y_test)
        _, vector_n_score = self.validate_vector_n_score(vector_n_score)

        n_score: int = len(vector_n_score) - 1
        score = self.coder_predict_score(x_test, vector_n_score, False)
        number_0_each_score: npt.NDArray[np.int_] = np.zeros(n_score, int)
        number_1_each_score: npt.NDArray[np.int_] = np.zeros(n_score, int)
        for id_data in range(len(y_test)):
            if y_test[id_data]:
                number_1_each_score[score[id_data] - 1] += 1
            else:
                number_0_each_score[score[id_data] - 1] += 1
        mortality: npt.NDArray[np.float_] = np.zeros(n_score)
        for s in range(n_score):
            if number_0_each_score[s] or number_1_each_score[s]:
                mortality[s] = (
                    100
                    * number_1_each_score[s]
                    / (number_0_each_score[s] + number_1_each_score[s])
                )
            else:
                mortality[s] = -1

        if print_mortality:
            for s in range(n_score):
                if mortality[s] == -1:
                    print(
                        f"Score={s+1}: There is not any data which matches "
                        f"this score!"
                    )
                else:
                    print(f"Score={s+1} is {mortality[s]:.2f}%")

        return mortality, number_0_each_score, number_1_each_score

    def coder_print_model(self) -> None:
        """Print the model data when the model has been fitted."""
        if self.model_fitted:
            self.print_binary_function_model()
        else:
            raise RuntimeError("Model has not been fitted yet!")


class Metric(Base):
    """Metrics to assess the performance of the model."""

    @staticmethod
    def set_confusion_matrix(
        y_test: Union[list[bool], npt.NDArray[np.bool_]],
        y_predicted: Union[list[bool], npt.NDArray[np.bool_]],
    ) -> tuple[float, float, float, float]:  # numpydoc ignore=RT03
        """
        Set confusion matrix.

        Parameters
        ----------
        y_test : ndarray of shape (n_samples,)
            Target output value in one_to_one mapping of binary input `x_test`.
        y_predicted : ndarray of shape (n_samples,)
            Predicted output value of the model in one_to_one mapping of
            binary input `x_test`.

        Returns
        -------
        accuracy : float
        recall : float
        precision : float
        F1 : float

        Notes
        -----
        confusion matrix:
                       Actual  Actual
                        0      1
        Predicted   0   TN     FN
        Predicted   1   FP     TP
        """
        # TODO validate y_test and y_predicted
        confusion_matrix: npt.NDArray[np.int_] = np.zeros((2, 2), int)

        for id_data in range(len(y_test)):
            if y_predicted[id_data] == y_test[id_data]:
                if y_predicted[id_data] == 0:
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[1][1] += 1
            else:
                if y_predicted[id_data] == 0:
                    confusion_matrix[0][1] += 1
                else:
                    confusion_matrix[1][0] += 1

        accuracy: float
        recall: float
        precision: float
        F1: float

        accuracy = float(
            confusion_matrix[0][0] + confusion_matrix[1][1]
        ) / float(confusion_matrix.sum())
        if confusion_matrix[1][1] + confusion_matrix[0][1] != 0:
            recall = confusion_matrix[1][1] / (
                confusion_matrix[1][1] + confusion_matrix[0][1]
            )
        else:
            recall = 1
        if confusion_matrix[1][1] + confusion_matrix[1][0] != 0:
            precision = confusion_matrix[1][1] / (
                confusion_matrix[1][1] + confusion_matrix[1][0]
            )
        else:
            precision = 1
        if (
            2 * confusion_matrix[1][1]
            + confusion_matrix[1][0]
            + confusion_matrix[0][1]
        ) != 0:
            F1 = (
                2
                * confusion_matrix[1][1]
                / (
                    2 * confusion_matrix[1][1]
                    + confusion_matrix[1][0]
                    + confusion_matrix[0][1]
                )
            )
        else:
            F1 = 1

        return accuracy, recall, precision, F1


class PseudoBooleanFunc(BasePseudoBooleanFunc):
    """
    Build Pseudo-Boolean function.

    Parameters
    ----------
    arity : int
        Arity of the Pseudo-Boolean function.

    Attributes
    ----------
    num : int
        A counter for filling the dict_new.

    dict_new : dict
        Stores members of each statement in the Pseudo-Boolean function.
        When `n=3` the Pseudo-Boolean function is `a0 + a1 X1 + a2 X2 + a3
        X3 + a4 X1 X2 + a5 X1 X3 + a6 X2 X3 + a7 X1 X2 X3`. So, 'dict_new'
        stores for key `0` an empty array as there is not any Xi for that.
        It stores for key `4` `[1,2]` which represents `X1 X2`.

    coefficient_matrix : ndarray of shape(2**n,2**n)
        To find the coefficients of the Pseudo-Boolean function in matrix
        form of `Ax=y`, matrix A should be built. `x` is a vector of unkown
        coefficients and `A` is a matrix in which each row represents a
        binary input. For instance, when `n=3`, each row of the `matrix` is
        filled as below:
        `[1, X1, X2, X3, X1*X2, X1*X3, X2*X3, X1*X2*X3]`.

    Examples
    --------
    >>> from noisecut.model.noisecut_coder import PseudoBooleanFunc
    >>> pseudo_boolean_func = PseudoBooleanFunc(3)
    >>> x = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1],\
     [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    >>> y = [0, 0, 1, 1, 0, 1, 0, 1]
    >>> coef = pseudo_boolean_func.get_coef_boolean_func(x, y)
    The Boolean Function is:
    0.00 + 0.00 X1 + 1.00 X2 + 0.00 X3 + 0.00 X1 X2 + 1.00 X1 X3 + -1.00 X2
    X3 + 0.00 X1 X2 X3 +
    >>> print(coef)
    [ 0.  0.  1.  0.  0.  1. -1.  0.]
    >>> pseudo_boolean_func.dict_new
    {0: array([], dtype=float64), 1: array([1]), 2: array([2]), 3: array([
    3]), 4: array([1, 2]), 5: array([1, 3]), 6: array([2, 3]), 7: array([1,
    2, 3])}
    """

    def __init__(self, arity: int) -> None:
        assert self.validate_arity(arity)
        self.num: int = 1
        self.dict_new: dict[int, npt.NDArray[np.int_]] = {0: np.zeros(0, int)}
        self.coefficient_matrix: npt.NDArray[np.float_] = np.ones(
            (2**self.arity, 2**self.arity)
        )

    def __recursion(self, n_size: int, level: int) -> None:
        """
        Fill a specific element of `dict_new`.

        Parameters
        ----------
        n_size : int
            Number of Xi in each statement of the Pseudo-Boolean function.

        level : int
            Number of loop in which the code will be run, it starts from 0.
        """
        if level == n_size - 1:
            for self.i_v[level] in range(  # noqa: B007, B020
                self.i_v[level - 1] + 1, self.arity + 1
            ):
                self.dict_new[self.num] = np.copy(self.i_v)
                self.num += 1

        else:
            for self.i_v[level] in range(  # noqa: B007, B020
                self.i_v[level - 1] + 1, self.arity + 1
            ):
                self.__recursion(n_size, level + 1)

    def __build(self) -> None:
        """Fill `dict_new` as an attribute of the class."""
        for n_size in range(self.arity):
            self.i_v = np.zeros(n_size + 1, int)
            self.__recursion(n_size + 1, 0)

    def __fill_coef_matrix(self, x: npt.NDArray[np.bool_]) -> None:
        """
        Fill `matrix` as an attribute of the class.

        Parameters
        ----------
        x : ndarray of bool of shape (2**n, n)
            Binary data input to set a Pseudo-Boolean function for.

        Notes
        -----
        For instance, when `n=3`, each row of the `matrix` is filled as below:
        `[1, X1, X2, X3, X1*X2, X1*X3, X2*X3, X1*X2*X3]`.
        """
        self.__build()
        for row in range(2**self.arity):
            for column in range(2**self.arity):
                for i in range(len(self.dict_new[column])):
                    self.coefficient_matrix[row][column] *= x[row][
                        self.dict_new[column][i] - 1
                    ]

    def get_coef_boolean_func(
        self,
        x: npt.NDArray[np.bool_],
        y: npt.NDArray[np.float_],
        print_allowance: bool = True,
    ) -> npt.NDArray[np.float_]:
        """
        Return the coefficient of the Pseudo-Boolean function.

        Parameters
        ----------
        x : ndarray of bool of shape (2**n, n)
            Binary data input to set a Pseudo-Boolean function for.

        y : ndarray of float of shape (2**n,)
            Data output in one_to_one mapping of binary data input `x`.

        print_allowance : bool
            Whether print the computed Pseudo-Boolean function.

        Returns
        -------
        ndarray of float of shape(2**n,)
            Csoefficient of the Pseudo-Boolean function.
        """
        _, x, y = self.validate_x_y(x, y)
        self.__fill_coef_matrix(x)

        coefficient = np.linalg.solve(self.coefficient_matrix, y)
        if print_allowance:
            print("The Boolean Function is:")
            for i in range(len(coefficient)):
                print(f"{coefficient[i]:.2f}", end=" ")
                for j in range(len(self.dict_new[i])):
                    print(f"X{self.dict_new[i][j]}", end=" ")
                print("+", end=" ")
            print("")
        return coefficient
