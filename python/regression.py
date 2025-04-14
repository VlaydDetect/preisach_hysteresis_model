import numpy as np


class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.predicted = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        self.predicted = X @ np.r_[self.intercept, self.coefficients]
        return self.predicted

    def r2(self, y_true):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - self.predicted) ** 2)
        return 1 - ss_residual / ss_total
