import random
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as st


class KNNClassifier():
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.__distances: list[list[float]] = [[]]

    def __str__(self) -> str:
        pass
    
    def plot(self, style: str = "fivethirtyeight") -> None:
        """
        Plots graph of dataset and the regression line.
        """
        st.use(style)
        try:
            pass
        except ValueError:
            print("\nERROR: No data to plot.\nPlease fit the model before plotting the line.\n\n")

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        train_set: dict[int, list[list[float]]] = {classNo: [] for classNo in y}
        test_set: dict[int, list[list[float]]] = {classNo: [] for classNo in y}
        for count, i in enumerate(y):
            train_set[i].append(X[count])
            test_set[i].append(X[count])
        # print(train_set, "\n\n\n\n\n\n\n\n\n\n", test_set,"\n\n\n\n\n\n\n\n", len(train_set[2]), len(train_set[4]), len(test_set[2]), len(test_set[4]))
        vote_list: list[int] = []
        for group in test_set:
            for predict_data in test_set[group]:
                vote_list.append(self.__k_nearest_neighbours__(train_set, predict_data))
    
    def __k_nearest_neighbours__(self, data: dict[int, list[list[float]]], predict: list[float]):
        if len(data) >= self.k:
            warnings.warn('K is set to a value less than total voting groups!')
        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])
        self.__distances.append(distances)
        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result: int = Counter(votes).most_common(1)[0][0]
        self.__confidence = Counter(votes).most_common(1)[0][1] / self.k
        return vote_result

    def predict(self, X: list[list[float]]):
        print(self.__distances)

    def confidence(self):
        return self.__confidence
    
