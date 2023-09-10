"""Useful methods to split or insert noise in dataset."""

# Author: Hedieh Mirzaieazar <hedieh.mirzaieazar@rwth-aachen.de>

import math
import random
from typing import (
    Any,
    Union,
)

import numpy as np
import numpy.typing as npt

from noisecut.tree_structured.base_structured_data import BaseStructuredData


class DataManipulator(BaseStructuredData):
    """
    DataManipulation class provides some useful methods to work with datasets.

    Attributes
    ----------
    x : {array-like, ndarray, dataframe} of shape (n_data, n_features)
        Each row of the array `x` is a binary input.

    y : {array-like, ndarray, dataframe} of shape (n_data,)
        Target output value in one_to_one mapping of binary input `x`.

    n_data: int
        Number of samples.
    """

    def __init__(self):
        super().__init__()
        self.x: npt.NDArray[np.bool_]
        self.y: npt.NDArray[np.bool_]
        self.n_data: int = 0

    def __set_xy(self, x: Any, y: Any) -> None:
        """
        Validate x and y, and set them to `x` and `y` attributes of the class.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, n_features)
        y : {array-like, ndarray, dataframe} of shape (n_data,)
        """
        status, self.x, self.y = self.validate_x_y(x, y)
        self.n_data = len(self.y)
        assert status

    @staticmethod
    def __generate_unique_random_numbers(
        start: int, end: int, num: int
    ) -> npt.NDArray[np.int_]:
        """
        Return an array of unique random numbers in range `start` and `end`.

        Parameters
        ----------
        start : int
            Min value of the random numbers.
        end : int
            Max value of the random numbers.
        num : int
            Count of random numbers to be generated.

        Returns
        -------
        ndarray of int of shape (num,)
            Unique random numbers array in range `start` and `end`.
        """
        arr: list[int] = []
        tmp: int = random.randint(start, end)
        for _ in range(num):
            while tmp in arr:
                tmp = random.randint(start, end)
            arr.append(tmp)
        arr.sort()
        return np.array(arr)

    def split_data(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
        percentage_training_data: float,
    ) -> tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
    ]:
        """
        Return training and test by splitting the dataset `x` and `y`.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, n_features)
            Binary input array.
        y : {array-like, ndarray, dataframe} of shape (n_data,)
            Target output value in one_to_one mapping of binary input `x`.
        percentage_training_data : float [0:100]
            Percentage of the presence of data in training dataset.

        Returns
        -------
        x_train : ndarray of bool of shape (n_training_data,n_features)
            Binary input array for train dataset.
        y_train : ndarray of bool of shape (n_training_data,)
            Target output value for train dataset.
        x_test : ndarray of bool of shape (n_test_data,n_features)
            Binary input array for test dataset.
        y_test : ndarray of bool of shape (n_test_data,)
            Target output value for test dataset.
        """
        status, percentage_training_data = self.is_percentage(
            percentage_training_data
        )
        assert status
        self.__set_xy(x, y)

        n_training_data: int = int(
            self.n_data * percentage_training_data / 100
        )
        rand_arr: npt.NDArray[np.int_] = self.__generate_unique_random_numbers(
            0, self.n_data - 1, n_training_data
        )

        x_training: npt.NDArray[np.bool_] = np.zeros(
            (n_training_data, self.dimension), bool
        )
        y_training: npt.NDArray[np.bool_] = np.zeros(n_training_data, bool)

        n_test_data: int = self.n_data - n_training_data
        x_test: npt.NDArray[np.bool_] = np.zeros(
            (n_test_data, self.dimension), bool
        )
        y_test: npt.NDArray[np.bool_] = np.zeros(n_test_data, bool)

        rand_num: int = rand_arr[0]
        train_id: int = 0
        test_id: int = 0
        for data_id in range(self.n_data):
            if rand_num == data_id:
                x_training[train_id, :] = self.x[rand_num, :]
                y_training[train_id] = self.y[rand_num]
                train_id += 1
                try:
                    rand_num = rand_arr[train_id]
                except IndexError:
                    rand_num = 0
            else:
                x_test[test_id, :] = self.x[data_id, :]
                y_test[test_id] = self.y[data_id]
                test_id += 1

        return x_training, y_training, x_test, y_test

    def get_noisy_data(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
        percentage_noise: float,
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """
        Return noisy dataset by adding noise in the given dataset `x` and `y`.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, n_features)
            Binary input.
        y : {array-like, ndarray, dataframe} of shape (n_data,)
            Target output value in one_to_one mapping of binary input `x`.
        percentage_noise : float [0:100]
            Percentage of the added noise in dataset.

        Returns
        -------
        x : ndarray of bool of shape (n_data, dimension)
            Noisy binary input.
        y : ndarray of bool of shape (n_data,)
            Noisy Target output.
        """
        status, percentage_noise = self.is_percentage(percentage_noise)
        assert status
        self.__set_xy(x, y)

        n_to_change: int = math.ceil(self.n_data * percentage_noise / 100)
        rand_arr: npt.NDArray[np.int_] = self.__generate_unique_random_numbers(
            0, self.n_data - 1, n_to_change
        )

        for i in range(n_to_change):
            if self.y[rand_arr[i]]:
                self.y[rand_arr[i]] = False
            else:
                self.y[rand_arr[i]] = True

        return self.x, self.y

    def write_data_in_file(
        self,
        x: Union[list[bool], npt.NDArray[np.bool_]],
        y: Union[list[bool], npt.NDArray[np.bool_]],
        file_name: str,
    ) -> None:
        """
        Write the input dataset `x` and `y` in a file with name `file_name`.

        Parameters
        ----------
        x : {array-like, ndarray, dataframe} of shape (n_data, n_features)
            Binary input.
        y :  {array-like, ndarray, dataframe} of shape (n_data,)
            Target output value in one_to_one mapping of binary input `x`.
        file_name : str
            Name of the file (with or without the path) in which you aim to
            store dataset `x` and `y`.
        """
        self.__set_xy(x, y)

        with open(file_name, "w") as file:
            for id_data in range(self.n_data):
                for d in range(self.dimension):
                    file.write(str(int(self.x[id_data][d])) + "    ")

                file.write(str(int(self.y[id_data])) + "\n")
