from __future__ import annotations

import numpy as np

from ..data.dataset import Dataset
from ..exceptions import ConfigError
from .base import BaseRecommender
from .registry import register_model


@register_model("mf")
class MatrixFactorizationRecommender(BaseRecommender):
    """Matrix factorisation trained with the BPR pairwise ranking loss (pure NumPy).

    score(u, i) = P[u] . Q[i] + b[i]
    """
    name = "mf"

    def __init__(self, n_factors: int = 32, learning_rate: float = 0.05,
                 reg: float = 0.01, n_epochs: int = 30, batch_size: int = 256,
                 seed: int = 42):
        super().__init__()
        self._check_int("n_factors", n_factors)
        self._check_int("n_epochs", n_epochs)
        self._check_int("batch_size", batch_size)
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) \
                or learning_rate <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {learning_rate!r}")
        if not isinstance(reg, (int, float)) or isinstance(reg, bool) or reg < 0:
            raise ConfigError(f"reg must be >= 0, got {reg!r}")
        self._n_factors, self._lr, self._reg = n_factors, float(learning_rate), float(reg)
        self._n_epochs, self._batch, self._seed = n_epochs, batch_size, seed
        self._P = self._Q = self._b = None
        self._has_history = None
        self.loss_history_ = []

    @staticmethod
    def _check_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError(f"{name} must be a positive integer, got {value!r}")

    def _fit(self, dataset: Dataset) -> None:
        rng = np.random.default_rng(self._seed)
        pos = dataset.positive_matrix.tocoo()
        users, items = pos.row, pos.col
        n_u, n_i, f = dataset.n_users, dataset.n_items, self._n_factors
        P = rng.normal(0, 0.1, (n_u, f))
        Q = rng.normal(0, 0.1, (n_i, f))
        b = np.zeros(n_i)
        keys = np.sort(users.astype(np.int64) * n_i + items)   # positive pairs, for neg. sampling
        self.loss_history_ = []

        for _ in range(self._n_epochs):
            perm = rng.permutation(len(users))
            total = 0.0
            for s in range(0, len(perm), self._batch):
                idx = perm[s:s + self._batch]
                u, i = users[idx], items[idx]
                j = rng.integers(0, n_i, size=len(idx))
                for _retry in range(3):                       # avoid sampling a positive as negative
                    k = u.astype(np.int64) * n_i + j
                    pos_pos = np.searchsorted(keys, k)
                    bad = (pos_pos < len(keys)) & (keys[np.minimum(pos_pos, len(keys) - 1)] == k)
                    if not bad.any():
                        break
                    j[bad] = rng.integers(0, n_i, size=int(bad.sum()))
                pu, qi, qj = P[u], Q[i], Q[j]
                x = (pu * (qi - qj)).sum(1) + b[i] - b[j]
                g = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))  # sigmoid(-x)
                total += float(np.log1p(np.exp(-np.clip(x, -30, 30))).sum())
                lr, reg = self._lr, self._reg
                np.add.at(P, u, lr * (g[:, None] * (qi - qj) - reg * pu))
                np.add.at(Q, i, lr * (g[:, None] * pu - reg * qi))
                np.add.at(Q, j, lr * (-g[:, None] * pu - reg * qj))
                np.add.at(b, i, lr * (g - reg * b[i]))
                np.add.at(b, j, lr * (-g - reg * b[j]))
            self.loss_history_.append(total / len(users))

        self._P, self._Q, self._b = P, Q, b
        self._has_history = np.diff(dataset.positive_matrix.indptr) > 0

    def _score_user(self, user_idx: int) -> np.ndarray:
        if not self._has_history[user_idx]:      # cold user: fall back to item bias
            return self._b.copy()
        return self._Q @ self._P[user_idx] + self._b
