import numpy as np

class MulticlassNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001, epochs=1000):
      self.lr = lr
      self.epochs = epochs
      self.output_size = output_size

      # Initialize weights
      self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
      self.b1 = np.zeros((1, hidden_size))
      self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
      self.b2 = np.zeros((1, output_size))

    def relu(self, x):
      return np.maximum(0, x)

    def relu_derivative(self, x):
      return (x > 0).astype(float)

    def softmax(self, x):
      # Clip values for numerical stability
      x = np.clip(x, -500, 500)
      exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
      return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, predictions, targets):
      epsilon = 1e-9  # To avoid log(0)
      return -np.mean(np.sum(targets * np.log(predictions + epsilon), axis=1))

    def _one_hot(self, y, num_classes):
      one_hot = np.zeros((len(y), num_classes))
      one_hot[np.arange(len(y)), y] = 1
      return one_hot

    def fit(self, X, y):
      y_one_hot = self._one_hot(y, self.output_size)

      for _ in range(self.epochs):
        # Forward pass
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        predictions = self.softmax(z2)

        # Backward pass
        # Output layer error
        output_error = predictions - y_one_hot
        dW2 = np.dot(a1.T, output_error)
        db2 = np.sum(output_error, axis=0, keepdims=True)

        # Hidden layer error
        hidden_error = np.dot(output_error, self.W2.T) * self.relu_derivative(z1)
        dW1 = np.dot(X.T, hidden_error)
        db1 = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases 
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
      z1 = np.dot(X, self.W1) + self.b1
      a1 = self.relu(z1)
      z2 = np.dot(a1, self.W2) + self.b2
      predictions = self.softmax(z2)
      return np.argmax(predictions, axis=1)