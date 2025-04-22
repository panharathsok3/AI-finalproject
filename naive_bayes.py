import numpy as np

class MulticlassNaiveBayes:
    def fit(self, X, y):
      self.classes = np.unique(y)

      # Initialize parameters
      self.mean = {}
      self.var = {}
      self.priors = {}

      for c in self.classes:
        X_c = X[y == c]
        self.mean[c] = np.mean(X_c, axis=0)
        self.var[c] = np.var(X_c, axis=0) + 1e-6  # prevent div by 0
        self.priors[c] = X_c.shape[0] / X.shape[0]

    def _log_likelihood(self, class_idx, x):
      mean = self.mean[class_idx]
      var = self.var[class_idx]
      numerator = -0.5 * ((x - mean) ** 2) / var
      denominator = -0.5 * np.log(2 * np.pi * var)
      return np.sum(numerator + denominator)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
      posteriors = []

      for c in self.classes:
        prior = np.log(self.priors[c])
        likelihood = self._log_likelihood(c, x)
        posterior = prior + likelihood
        posteriors.append(posterior)

      return self.classes[np.argmax(posteriors)]
