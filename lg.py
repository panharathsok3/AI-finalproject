import numpy as np

# Implementation of Multiclass Naive Bayes classifier.
class MulticlassLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    # Softmax function to convert logits to probabilities
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Convert labels to one-hot encoding
    def _one_hot(self, y, n_classes):
        onehot = np.zeros((len(y), n_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot

     # Fit the model to the training data
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        y_onehot = self._one_hot(y, n_classes)

        # Gradient descent
        for _ in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            probs = self._softmax(logits)

            error = probs - y_onehot

            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error, axis=0) / n_samples

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    # Predict the class labels for the input data
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)