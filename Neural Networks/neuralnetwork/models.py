import numpy as np
from .layers import Layer
from . import losses


class Sequential:
    def __init__(self, layers: list[Layer], loss: losses.Loss = losses.CategoricalCrossentropy) -> None:
        self.layers = layers
        self.loss_function = loss
        self.loss = 0
        self.__output = []
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        for count, layer in enumerate(self.layers):
            if count == 0:
                layer.forward(X)
                activation = layer.activation
                self.__output = activation.forward(activation, layer.output)
                continue
            else:
                layer.forward(self.__output)
                activation = layer.activation
                self.__output = activation.forward(activation, layer.output)
        
        self.loss = self.loss_function.calculate(self.loss_function, self.__output, y)
        print(self.__output)
