import numpy as np


class K_Means:
    def __init__(self, k: int = 2, tolerence = 0.001, max_itr = 300):
        """Class for K-Means Clustering

        Args:
            - k (int, optional): total cluster numbers. Defaults to 2.
            - tolerence (float, optional): tolerence to centroids movement. Defaults to 0.001.
            - max_itr (int, optional): maximum iterations till which clustering is done. Defaults to 300.
        """
        self.k = k
        self.tol = tolerence
        self.max_itr = max_itr

    def fit(self, X, y = None) -> list:
        """Fit method for training the model.

        Args:
            - X (arrayLike, list, npArray): data for training.
            - y (arrayLike, list, npArray, optional): For consistency of API. Even if inserted, it is ignored. Defaults to None.
        """
        self.centroids = {}
        self.__moves_history = []
        for i in range(self.k):
            self.centroids[i] = X[i]

        for i in range(self.max_itr):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid  = self.centroids[c]
                move = np.sum((current_centroid-original_centroid) / original_centroid*100.0)
                if move > self.tol:
                    self.__moves_history.append(move)
                    optimized = False

            if optimized:
                break
        
        return self.__moves_history

    def predict(self, X):
        """Predict method for predicting values.

        Args:
            - X (arrayLike, list, npArray): input data.
        """
        predicted = []
        for point in X:
            distances = [np.linalg.norm(point-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            predicted.append(classification)
        return predicted
