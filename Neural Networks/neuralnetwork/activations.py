import numpy as np

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output