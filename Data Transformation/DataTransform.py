import numpy as np

class MinMaxScaler:
    def __init__(self) -> None:
        self.old_min = 0
        self.old_max = 0
        self.new_min = 0
        self.new_max = 1
        self.__fit = False
    
    def fit(self, data: np.ndarray, min: float = 0.0, max : float = 1.0) -> None:
        self.data = np.array(data)
        self.new_min = min
        self.new_max = max

        self.old_min = np.amin(self.data)
        self.old_max = np.amax(self.data)
        self.__fit = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.__fit == True:
            data = np.array(data)
            data = (( (data - self.old_min) / (self.old_max - self.old_min) ) * (self.new_max - self.new_min) ) + self.new_min
            return np.array(data)
        else:
            raise(Exception("The scaler is not fitted on the data!"))
    
    def fit_transform(self, data: np.ndarray, min: float = 0.0, max : float = 1.0):
        self.data = np.array(data)
        self.new_min = min
        self.new_max = max

        self.old_min = np.amin(self.data)
        self.old_max = np.amax(self.data)

        self.data = np.array(self.data)
        self.data = (( (self.data - self.old_min) / (self.old_max - self.old_min) ) * (self.new_max - self.new_min) ) + self.new_min
        return np.array(self.data)
    
class StandardScaler:
    def __init__(self) -> None:
        self.standard_deviation = 0
        self.mean = 0.5
        self.__fit = False

    def fit(self, data: np.ndarray | list | tuple) -> None:
        self.data = np.array(data)

        self.standard_deviation = np.std(self.data)
        self.mean = np.mean(self.data)
        self.__fit = True
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.__fit == True:
            data = np.array(data)
            self.z_score = (data - self.mean) / self.standard_deviation
            return np.array(self.z_score)
        else:
            raise(Exception("The scaler is not fitted on the data!"))

    def fit_transform(self, data: np.ndarray):
        self.data = np.array(data)

        self.standard_deviation = np.std(self.data)
        self.mean = np.mean(self.data)

        self.z_score = (self.data - self.mean) / self.standard_deviation
        return np.array(self.z_score)