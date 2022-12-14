import numpy as np
from . import activations

class Layer:
    def __init__(self, n_inputs, n_neurons, activation = activations.ReLU()) -> None:
        self.weights = []
        self.dweights = []
        self.biases = []
        self.dbiases = []
        self.dinputs = []
        self.activation = activation
        self.output = []
        self.__n_inputs = n_inputs
        self.__n_neurons = n_neurons
        
    @property
    def shape(self):
        return (self.__n_inputs, self.__n_neurons)
    
    def forward(self, inputs):
        pass
    
    def backward(self, dvalues):
        pass

class Layer_Dense(Layer):

    def __init__(self, n_inputs, n_neurons, activation):
        self.activation = activation
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
