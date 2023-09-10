"""
NoiseCut class for the usage of user.
"""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>
from typing import Union

import numpy as np
import numpy.typing as npt

from noisecut.model.noisecut_coder import CoderNoiseCut


class NoiseCut:
    """
    NoiseCut class for implementing NoiseCut method.

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
    """

    def __init__(
        self,
        n_input_each_box: Union[list[int], npt.NDArray[np.int_]],
        threshold: float = 0.5,
    ) -> None:
        self.__model = CoderNoiseCut(n_input_each_box, threshold)

    def fit(
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
        self.__model.coder_fit(
            x, y, with_more_data, print_result, print_weights
        )

    def predict(
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
        return self.__model.coder_predict(x)

    def predict_all(self) -> npt.NDArray[np.bool_]:
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
        return self.__model.coder_predict_all()

    def set_uncertainty_measure(
        self, file_path_result: Union[str, None] = None
    ) -> None:
        """
        Set uncertainty.

        Parameters
        ----------
        file_path_result : str, default=None
            Path of a file to save the uncertainty result to it.
        """
        self.__model.coder_set_uncertainty_measure(file_path_result)

    def predict_probability_of_being_1(
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
        return self.__model.coder_predict_probability_of_being_1(x)

    def predict_pseudo_boolean_func_coef(self) -> npt.NDArray[np.float_]:
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
        return self.__model.coder_predict_pseudo_boolean_func_coef()

    def predict_score(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        vector_n_score: Union[list[float], npt.NDArray[np.float_]],
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

        Returns
        -------
        ndarray of int of shape (n_samples,)
            Score array is set in one_to_one mapping of binary input `x`.
        """
        return self.__model.coder_predict_score(x, vector_n_score)

    def predict_mortality_of_each_score(
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
        return self.__model.coder_predict_mortality_of_each_score(
            x_test, y_test, vector_n_score, print_mortality
        )

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
        return self.__model.get_binary_function_of_box(id_box)

    def get_binary_function_black_box(self) -> npt.NDArray[np.bool_]:
        """
        Return binary function of the 2nd-layer black box.

        Returns
        -------
        ndarray of bool of shape (2**n_box,)
            Binary function of the output box.
        """
        return self.__model.get_binary_function_black_box()

    def print_model(self) -> None:
        """Print the model data when the model has been fitted."""
        self.__model.coder_print_model()
