from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..data.dataset import Dataset
from .base import BaseRecommender, validate_k
from .registry import register_model


@register_model("popularity")
class PopularityRecommender(BaseRecommender):
    """Scores each item by its number of positive interactions (non-personalised)."""
    name = "popularity"

    def __init__(self):
        super().__init__()
        self._counts = np.zeros(0)

    def _fit(self, dataset: Dataset) -> None:
        self._counts = np.asarray(dataset.positive_matrix.sum(axis=0)).ravel()

    def _score_user(self, user_idx: int) -> np.ndarray:
        return self._counts

    def top_popular(self, k: int = 10) -> List[Tuple[str, float]]:
        """Top-k overall; useful as a cold-start fallback for unknown users."""
        self._require_fitted()
        k = validate_k(k)
        order = np.argsort(-self._counts, kind="stable")[:k]
        return [(self._dataset.item_id(int(j)), float(self._counts[j])) for j in order]
