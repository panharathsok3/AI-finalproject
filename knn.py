import numpy as np
from collections import Counter

# Implementation of the KNN algorithm for classification tasks.
class knn:
  def __init__(self, k=3):
    self.k = k

   # Fit the model to the training data
  def fit(self, X, y):
    self.X_train = np.array(X)
    self.y_train = np.array(y)

  # Compute the Euclidean distance between two points
  def _euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

  # Predict the class of a sample
  def predict(self, X):
    X = np.array(X)
    predictions = [self._predict(x) for x in X]
    return np.array(predictions)
    
  # Helper function to predict the class of a single sample
  def _predict(self, x):
    # Compute distances to all training samples
    distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

    # Get k nearest samples
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]

    # Majority vote
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]
