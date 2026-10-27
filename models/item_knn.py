"""Item-based cosine CF, with overlap shrinkage and sparse top neighbours."""
import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, k=100, shrink=10):
        self.k, self.shrink = k, shrink

    def fit(self, history, **kwargs):
        x = history.astype(np.float64)
        gram = (x.T @ x).toarray()
        norm = np.sqrt(np.diag(gram))
        sim = gram / np.maximum(norm[:, None] * norm[None, :], 1e-12)
        sim *= gram / (gram + self.shrink + 1e-12)
        np.fill_diagonal(sim, 0)
        # Each source item contributes its k nearest target items.
        k = min(self.k, len(sim) - 1)
        ids = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(len(sim)), k)
        self.weights = sparse.csr_matrix(
            (sim[rows, ids.ravel()], (rows, ids.ravel())), shape=sim.shape)
        self.history = x
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
