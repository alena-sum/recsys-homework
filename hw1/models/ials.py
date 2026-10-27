import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.linalg.blas import dgemm, dgemv


class IALS:
    def __init__(self, factors=24, reg=1.0, alpha=20., epochs=8, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.epochs, self.seed = epochs, seed

    def targets(self, events, cutoff):
        return (events.rating.to_numpy() >= 4).astype(float)

    def fit(self, history, events, cutoff, **kwargs):
        n_users, n_items = history.shape
        targets = self.targets(events, cutoff)
        observed = sparse.csr_matrix((np.ones(len(events)),
            (events.user_idx, events.item_idx)), shape=history.shape)
        p = sparse.csr_matrix((targets,
            (events.user_idx, events.item_idx)), shape=history.shape)
        assert np.array_equal(observed.indices, p.indices)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .05, (n_users, self.factors))
        self.item_factors = rng.normal(0, .05, (n_items, self.factors))
        for _ in range(self.epochs):
            self.user_factors = self._step(self.item_factors, observed, p)
            self.item_factors = self._step(self.user_factors,
                                          observed.T.tocsr(), p.T.tocsr())
        return self

    def _step(self, fixed, observed, targets):
        base = dgemm(1., fixed, fixed, trans_a=True) + self.reg * np.eye(self.factors)
        out = np.zeros((observed.shape[0], self.factors))
        for u in range(observed.shape[0]):
            a, b = observed.indptr[u:u + 2]
            if a == b:
                continue
            y = fixed[observed.indices[a:b]]
            lhs = base + self.alpha * dgemm(1., y, y, trans_a=True)
            rhs = (1 + self.alpha) * dgemv(1., y, targets.data[a:b], trans=True)
            out[u] = solve(lhs, rhs, assume_a='pos', check_finite=False)
        return out

    def score(self, users):
        return dgemm(1., self.user_factors[users], self.item_factors, trans_b=True)
