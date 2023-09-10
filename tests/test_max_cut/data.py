from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = str(Path(__file__).parent)


class Data:
    def __init__(self):
        self.nVertices = None
        self.weight = None
        self.true_obj = None

    def get_weight(self):
        return self.weight

    def get_n_vertices(self):
        return self.nVertices

    def initialize_data(self, example):
        # sys.path.append('tests/max_cut')
        path = THIS_DIR + "/local_solver/instances/"
        dim = pd.read_csv(path + example, sep=" ", header=None, nrows=1)
        df = pd.read_csv(path + example, sep=" ", header=None, skiprows=1)

        self.nVertices = dim[0][0]
        nEdge = dim[1][0]

        self.weight = np.zeros((self.nVertices, self.nVertices))
        for k in range(nEdge):
            i = df.iloc[k, 0] - 1
            j = df.iloc[k, 1] - 1
            self.weight[i][j] = df.iloc[k, 2]
            self.weight[j][i] = df.iloc[k, 2]

        path_output = THIS_DIR + "/local_solver/output_of_instances/"
        dim = pd.read_csv(path_output + example, sep=" ", header=None, nrows=1)
        self.true_obj = dim[0][0]
