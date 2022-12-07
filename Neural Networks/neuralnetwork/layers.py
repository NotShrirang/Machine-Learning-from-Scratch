import numpy as np
from . import activations

class Layer:
    def __init__(self, n_inputs, n_neurons, activation = activations.ReLU) -> None:
        self.weights = []
        self.biases = []
        self.activation = activation
        self.output = []
        self.__n_inputs = n_inputs
        self.__n_neurons = n_neurons
        
    @property
    def shape(self):
        return (self.__n_inputs, self.__n_neurons)
    
    def forward():
        pass

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation = activations.ReLU) -> None:
        super().__init__(n_inputs, n_neurons, activation)
        np.random.seed(0)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases