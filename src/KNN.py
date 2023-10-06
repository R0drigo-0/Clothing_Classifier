import numpy as np
import math
import operator
from scipy.spatial.distance import cdist, cityblock


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.alt_dist = False  # True = MH | False = EUC

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: P x M x N x '3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        image_num = train_data.shape[0]
        if len(train_data.shape) == 3:
            self.train_data = train_data.reshape(image_num, -1).astype(np.float64)
        else:
            self.train_data = (
                np.mean(train_data, axis=3).reshape(image_num, -1).astype(np.float64)
            )

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        self.test_data = test_data
        if len(self.test_data.shape) == 3:
            self.test_data = self.test_data.reshape(self.test_data.shape[0], -1).astype(
                np.float64
            )
        else:
            self.test_data = (
                np.mean(self.test_data, axis=3)
                .reshape(self.test_data.shape[0], -1)
                .astype(np.float64)
            )

        if self.alt_dist == True:
            distance = cdist(self.test_data, self.train_data, cityblock)
        else:
            distance = cdist(self.test_data, self.train_data)

        index = []
        for i in distance:
            x = i.argsort()[:k]
            index.append(self.labels[x])

        self.neighbors = np.array(index)
        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        class_sorted = []
        for i in self.neighbors:
            value, index, rep = np.unique(i, return_index=True, return_counts=True)
            value = value[np.argsort(index)]
            rep = rep[np.argsort(index)]
            class_sorted.append(value[np.argmax(rep)])

        class_sorted = np.array(class_sorted, dtype="<U8")
        return class_sorted

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
