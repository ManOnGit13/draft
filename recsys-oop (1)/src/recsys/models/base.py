"""Abstract recommender. Public methods are concrete (Template Method);
subclasses only implement _fit() and _score_user()."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data.dataset import Dataset
from ..exceptions import DataError, InvalidParameterError, ModelNotFittedError


def validate_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise InvalidParameterError(f"k must be an integer, got {type(k).__name__}")
    if k <= 0:
        raise InvalidParameterError(f"k must be > 0, got {k}")
    return int(k)


class BaseRecommender(ABC):
    name = "base"

    def __init__(self):
        self._dataset: Optional[Dataset] = None

    # ---- state ----
    @property
    def is_fitted(self) -> bool:
        return self._dataset is not None

    @property
    def dataset(self) -> Dataset:
        self._require_fitted()
        return self._dataset  # type: ignore[return-value]

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted before use")

    # ---- hooks for subclasses ----
    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        """Learn parameters from the dataset."""

    @abstractmethod
    def _score_user(self, user_idx: int) -> np.ndarray:
        """Return a float array of length n_items with the score of every item."""

    # ---- public API ----
    def fit(self, dataset: Dataset) -> "BaseRecommender":
        if not isinstance(dataset, Dataset):
            raise DataError("fit() expects a Dataset")
        if dataset.positive_matrix.nnz == 0:
            raise DataError("no positive interactions (check rating_threshold)")
        self._dataset = None            # stay "unfitted" if _fit fails midway
        self._fit(dataset)
        self._dataset = dataset
        return self

    def score(self, user_id: str, item_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        """Scores for the given items (all items when item_ids is None)."""
        self._require_fitted()
        ds = self._dataset
        scores = self._score_user(ds.user_index(user_id))   # raises UnknownUserError
        if item_ids is None:
            return scores.copy()
        idx = [ds.item_index(i) for i in item_ids]          # raises UnknownItemError
        return scores[idx]

    def rank(self, user_id: str, exclude_seen: bool = True) -> np.ndarray:
        """Item indices ordered best-first (ties broken by item index)."""
        self._require_fitted()
        ds = self._dataset
        u = ds.user_index(user_id)
        scores = np.array(self._score_user(u), dtype=np.float64)
        n_candidates = ds.n_items
        if exclude_seen:
            seen = ds.seen_item_indices(u)
            scores[seen] = -np.inf
            n_candidates -= len(seen)
        return np.argsort(-scores, kind="stable")[:n_candidates]

    def recommend(self, user_id: str, k: int = 10,
                  exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Top-k (item_id, score). Returns fewer than k if not enough candidates."""
        self._require_fitted()
        k = validate_k(k)
        order = self.rank(user_id, exclude_seen)[:k]
        scores = self._score_user(self._dataset.user_index(user_id))
        return [(self._dataset.item_id(int(j)), float(scores[j])) for j in order]
