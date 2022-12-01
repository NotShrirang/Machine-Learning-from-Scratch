import numpy as np
from . import activations

class Layer:
    def __init__(self, activation = activations.ReLU) -> None:
        self.activation = activation
        self.output = []
    def forward():
        pass

class Layer_Dense(Layer):
    def __init__(self, n_inputs, n_neurons, activation = activations.ReLU) -> None:
        np.random.seed(0)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases