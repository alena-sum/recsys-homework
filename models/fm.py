"""Second-order FM trained with a manually differentiated BPR objective.

No recommender, optimizer, or trainable algorithm library is used.
Feature vectors are sparse binary multi-hot vectors. Mini-batch Adagrad sums
repeated feature gradients with np.add.at and updates its accumulators here.
"""
import numpy as np
from scipy.special import expit
from scipy.linalg.blas import dgemm


class FM:
    def __init__(self, factors=16, lr=.04, reg=.001, epochs=10,
                 batch_size=2048, seed=42):
        self.factors, self.lr, self.reg = factors, lr, reg
        self.epochs, self.batch_size = epochs, batch_size
        self.seed = seed

    def fit(self, history, events, item_features, **kwargs):
        self.n_users, self.n_items = history.shape
        ids = np.arange(self.n_items)[:, None]
        self.item_features = np.column_stack([ids, item_features])
        self.item_features = self.item_features + self.n_users
        n_features = int(self.item_features.max()) + 1
        rng = np.random.default_rng(self.seed)
        self.v = rng.normal(0, .05, (n_features, self.factors))
        self.w = np.zeros(n_features)
        av = np.full_like(self.v, 1e-8)
        aw = np.full_like(self.w, 1e-8)
        positive = events[events.rating >= 4]
        uu = positive.user_idx.to_numpy()
        ii = positive.item_idx.to_numpy()
        seen = [set() for _ in range(self.n_users)]
        for u, i in zip(events.user_idx, events.item_idx):
            seen[u].add(i)
        eligible = np.array([len(s) < self.n_items for s in seen])
        keep = eligible[uu]
        uu, ii = uu[keep], ii[keep]
        self.losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(uu))
            epoch_loss = 0.
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                u, i = uu[idx], ii[idx]
                j = rng.integers(self.n_items, size=len(idx))
                bad = np.array([b in seen[a] for a, b in zip(u, j)])
                while bad.any():
                    j[bad] = rng.integers(self.n_items, size=bad.sum())
                    bad = np.array([b in seen[a] for a, b in zip(u, j)])
                xp = np.column_stack([u, self.item_features[i]])
                xn = np.column_stack([u, self.item_features[j]])
                sp, vp, sump = self._forward(xp)
                sn, vn, sumn = self._forward(xn)
                delta = sp - sn
                epoch_loss += np.logaddexp(0, -delta).sum()
                g = -expit(-delta) / len(idx)
                # Compute both gradients BEFORE mutating weights.
                gvp = g[:, None, None] * (sump[:, None, :] - vp)
                gvn = -g[:, None, None] * (sumn[:, None, :] - vn)
                gvp += self.reg * vp / len(idx)
                gvn += self.reg * vn / len(idx)
                gwp = np.broadcast_to(g[:, None], xp.shape) + self.reg * self.w[xp] / len(idx)
                gwn = np.broadcast_to(-g[:, None], xn.shape) + self.reg * self.w[xn] / len(idx)
                # Sparse Adagrad, implemented here rather than a library optimizer.
                all_ids = np.r_[xp.ravel(), xn.ravel()]
                touched, inverse = np.unique(all_ids, return_inverse=True)
                gv = np.zeros((len(touched), self.factors))
                gw = np.zeros(len(touched))
                np.add.at(gv, inverse, np.concatenate([
                    gvp.reshape(-1, self.factors), gvn.reshape(-1, self.factors)]))
                np.add.at(gw, inverse, np.r_[gwp.ravel(), gwn.ravel()])
                av[touched] += gv ** 2
                aw[touched] += gw ** 2
                self.v[touched] -= self.lr * gv / np.sqrt(av[touched])
                self.w[touched] -= self.lr * gw / np.sqrt(aw[touched])
            self.losses.append(float(epoch_loss / max(len(uu), 1)))
        f = self.item_features
        iv = self.v[f]
        self.item_sum = iv.sum(axis=1)
        self.item_bias = self.w[f].sum(axis=1) + .5 * (
            self.item_sum ** 2 - (iv ** 2).sum(axis=1)).sum(axis=1)
        return self

    def _forward(self, x):
        v = self.v[x]
        total = v.sum(axis=1)
        score = self.w[x].sum(axis=1) + .5 * (
            total ** 2 - (v ** 2).sum(axis=1)).sum(axis=1)
        return score, v, total

    def score(self, users):
        users = np.asarray(users)
        return dgemm(1., self.v[users], self.item_sum, trans_b=True) + self.item_bias[None, :] + self.w[users, None]
