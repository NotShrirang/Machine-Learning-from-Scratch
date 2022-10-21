import warnings
from collections import Counter
import numpy as np


class KNNClassifier():
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.__distances: list[list[float]] = [[]]

    def __str__(self) -> str:
        print(self.__X__, self.y if self.__X__ is not None else "Not data to show.")

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        test_set: dict[int, list[list[float]]] = {classNo: [] for classNo in y}
        train_set: dict[int, list[list[float]]] = {classNo: [] for classNo in y}
        for count, i in enumerate(y):
            train_set[i].append(X[count])
            test_set[i].append(X[count])
        self.__X__ = train_set
        self.y = train_set
        vote_list: list[int] = []
        for group in test_set:
            for predict_data in test_set[group]:
                vote_list.append(self.__k_nearest_neighbours__(train_set, predict_data))
    
    def __k_nearest_neighbours__(self, data: dict[int, list[list[float]]], predict: list[float]):
        if len(data) >= self.k:
            warnings.warn('K is less than number of labels!')
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
        predicted_values = [self.__k_nearest_neighbours__(self.__X__, datapoint) for datapoint in X]
        return predicted_values

    def confidence(self):
        return self.__confidence
