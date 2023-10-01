import pandas as pd
import numpy as np
import tensorflow as tf


class data_loader:
    def __init__(self, path_content, path_cite) -> None:
        self.path_cite = path_cite
        self.path_content = path_content
        self.raw_data = pd.read_csv(path_content, sep='\t', header=None)
        num = self.raw_data.shape[0]
        index = self.raw_data.index
        firstdata = self.raw_data[0]
        c = zip(firstdata, index)
        map=dict(c)
        self.features = self.raw_data.iloc[:, 1:-1]

        self.labels = pd.get_dummies(self.raw_data[1434])
        raw_data_cites = pd.read_csv(path_cite, sep='\t', header=None)
        self.matrix = np.zeros((num, num))
        for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
            x, y = map[i], map[j]
            self.matrix[x][y] = self.matrix[y][x] = 1
        tem_d = np.sum(self.matrix, axis=1)
        self.d = np.zeros((num, num))
        for i in range(num):
            self.d[i][i] = 1.0/np.sqrt(tem_d[i])
        self.support = tf.matmul(tf.matmul(self.d, self.matrix), self.d)

    def get_data(self):
        return np.array(self.features), np.array(self.labels), np.array(self.support)