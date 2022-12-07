import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(self, output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    def forward(self, y_pred, y_true):
        pass

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: ## Incase user passes one-hot encoded vector.
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods