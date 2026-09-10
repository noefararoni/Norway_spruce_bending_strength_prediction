"""One gradient-descent implementation for both regression variants."""

import numpy as np


class LinearRegression:
    """MSE regression with optional soft coefficient-sign penalties.

    The penalty is summed, matching Last_drow_PI_LR.py. Width and Height
    are unconstrained. Inputs must be standardized using training data.
    """

    def __init__(
        self,
        learning_rate=0.1,
        n_iterations=120750,
        physics_lambda=0.0,
        positive_features=(),
        negative_features=(),
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError("n_iterations must be a positive integer.")
        if not np.isfinite(physics_lambda) or physics_lambda < 0:
            raise ValueError("physics_lambda must be finite and nonnegative.")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.physics_lambda = physics_lambda
        self.positive_features = tuple(positive_features)
        self.negative_features = tuple(negative_features)
        self.weights = None
        self.bias = None

    def fit(self, X, y, feature_names):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not X.size:
            raise ValueError("X must be a nonempty matrix and y a matching vector.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must contain only finite values.")
        names = list(feature_names)
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise ValueError("Feature names must be unique and match X columns.")
        positive, negative = set(self.positive_features), set(self.negative_features)
        if positive & negative or (positive | negative) - set(names):
            raise ValueError("Constraints must reference existing, nonoverlapping features.")
        signs = np.array([1 if n in positive else -1 if n in negative else 0 for n in names])
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            error = np.dot(X, self.weights) + self.bias - y
            gradient = (2.0 / len(y)) * np.dot(X.T, error)
            violations = self.weights * signs < 0
            gradient += 2 * self.physics_lambda * self.weights * violations
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * ((2.0 / len(y)) * np.sum(error))
            if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
                raise FloatingPointError("Training diverged; reduce the learning rate.")
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Fit the model before predicting.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights) or not np.isfinite(X).all():
            raise ValueError("Prediction data must be finite and match the fitted features.")
        return X @ self.weights + self.bias
