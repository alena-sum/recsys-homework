"""EASE с нулевой диагональю матрицы весов."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class EASE:
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, history, **kwargs):
        self.history = history.astype(np.float64)
        g = (self.history.T @ self.history).toarray()
        g.flat[::len(g) + 1] += self.reg
        p = cho_solve(cho_factor(g), np.eye(len(g)), check_finite=False)
        self.weights = -p / np.diag(p)[None, :]
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights
