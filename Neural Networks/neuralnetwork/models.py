import numpy as np
from .layers import Layer
from . import losses
import os
import datetime

class Sequential:
    def __init__(self, layers: list[Layer], loss: losses.Loss = losses.CategoricalCrossentropy) -> None:
        self.layers = layers
        self.loss_function = loss
        self.__loss = 0
        self.__least_loss = 99999
        self.__output = []
        self.__best_weights = []
        self.__best_biases = []
    
    @property
    def model_loss(self):
        return self.__least_loss

    def fit(self, X, y, epoch: int, print_output: bool = False, iteration: int = 5000):
        self.iteration = 5000
        self.__best_weights = [layer.weights for layer in self.layers]
        self.__best_biases = [layer.biases for layer in self.layers]
        history = []
        for epoch_number in range(epoch):
            os.system('cls')
            print("Epoch", epoch_number+1, ":")
            for i in range(self.iteration):
                self.__current_weights = []
                self.__current_biases  = []
                weights_list = []
                bias_list = []
                for count, layer in enumerate(self.layers):
                    if count == 0:
                        layer.forward(X)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                    else:
                        layer.weights = layer.weights + 0.05 * np.random.randn(layer.shape[0], layer.shape[1])
                        layer.biases = layer.biases + 0.05 * np.random.randn(1, layer.shape[1])
                        layer.forward(self.__output)
                        activation = layer.activation
                        self.__output = activation.forward(activation, layer.output)
                    weights_list.append(layer.weights)
                    bias_list.append(layer.biases)
                self.__current_weights.append(weights_list)
                self.__current_biases.append(bias_list)
                loss_function = self.loss_function
                self.__loss = loss_function.calculate(loss_function, self.__output, y)
                predictions = np.argmax(self.__output, axis=1)
                accuracy = np.mean(predictions==y)
                if self.__loss < self.__least_loss:
                    if print_output:
                        perc_done = i / self.iteration
                        eq_count = int(perc_done * 20)
                        eq_left = 19 - eq_count
                        my_str = f"Epoch {epoch_number+1}/{epoch}" + "[" + "="*eq_count + ">" + " "*eq_left + "] " + f"loss = {self.__loss :.4f}, {accuracy = :.4f}"
                        os.system('cls')
                        print(my_str)
                    self.__best_weights = self.__current_weights.copy()
                    self.__best_biases = self.__current_biases.copy()
                    self.__least_loss = self.__loss
                else:
                    for count, layer in enumerate(self.layers):
                        layer.weights = self.__best_weights[0][count]
                        layer.biases = self.__best_biases[0][count]
            print("loss: ", self.__least_loss)
            history.append(self.__least_loss)
        return history

    def predict(self, X):
        predictions = []
        for x_point in X:
            for count, layer in enumerate(self.layers):
                if count == 0:
                    layer.forward(x_point)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
                else:
                    layer.weights = layer.weights + 0.05 * np.random.randn(layer.shape[0], layer.shape[1])
                    layer.biases = layer.biases + 0.05 * np.random.randn(1, layer.shape[1])
                    layer.forward(self.__output)
                    activation = layer.activation
                    self.__output = activation.forward(activation, layer.output)
            predictions.append(self.__output)
        return predictions

    def save(self, name: str = f"my_model_{datetime.datetime.now()}"):
        if not os.path.exists(f"./saved_models/{name}"):
            os.mkdir("saved_models")
            os.mkdir(f"./saved_models/{name}")
            np.save(f"./saved_models/{name}/model_weights", self.__best_weights)
            np.save(f"./saved_models/{name}/model_biases", self.__best_biases)
        import pickle
        with open(f"./saved_models/{name}/model.nn", "wb") as f: 
            pickle.dump(self, f)
        return True   
    
def load_model(path_to_model: str) -> Sequential:
    if os.path.exists(f"./saved_models/{path_to_model}"):
        import pickle
        with open(f"./saved_models/{path_to_model}/model.nn", "rb") as f: 
            model = pickle.load(f)
        return model
    else:
        raise FileNotFoundError(f"Model {path_to_model} not found.")
