from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as st


class Linear_Regression():
    """
    Class for linear regression.
    """
    def __init__(self) -> None:
        self.__xs, self.__ys, self.__slope, self.__y_intercept, self.__regression_line = None, None, None, None, None
        self.__predict_x, self.__predict_y = 0, 0

    def __str__(self) -> str:
        try:
            return str([(self.__xs[i], self.__regression_line[i]) for i in range(len(self.__regression_line))])
        except TypeError:
            print("ERROR: Model has no data.\nPlease fit the model before printing it.\n")
            return ""

    def __getitem__(self, item):
        return self.__regression_line[item]

    def accuracy(self):
        return self.coefficient_of_determination_sqaured()*100

    def plot(self, style: str = "fivethirtyeight") -> None:
        """
        Plots graph of dataset and the regression line.
        """
        st.use(style)
        try:
            plt.scatter(self.__xs, self.__ys)
            plt.scatter(self.__predict_x, self.__predict_y, s=100, color='r')
            plt.plot(self.__xs, self.__regression_line)
            plt.show()
        except ValueError:
            print("\nERROR: No data to plot.\nPlease fit the model before plotting the line.\n\n")

    def fit(self, X, y):
        m = ( ( (mean(X) * mean(y)) - (mean(X*y)) ) /
              ( (mean(X)**2) - (mean(X**2) ) ) )

        b = mean(y) - m*mean(X)
        self.__xs = X
        self.__ys = y
        self.__slope = m
        self.__y_intercept = b
        self.__regression_line = [ (m*x_cord)+b for x_cord in X ]

    def predict(self, X: list | float):
        self.__predict_x = X
        self.__predict_y = np.array([(self.__slope*x_cord)+self.__y_intercept for x_cord in X], dtype=np.float64)
        return self.__predict_y

    def coefficient_of_determination_sqaured(self) -> float:
        squared_error = sum((self.__ys - self.__regression_line)**2)
        y_mean_line = [mean(self.__ys) for y in self.__regression_line]
        squared_error_regr = squared_error
        squared_error_y_mean = sum((self.__ys - y_mean_line)**2)
        return 1 - (squared_error_regr / squared_error_y_mean)
