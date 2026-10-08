"""Ranking metrics. Each metric scores ONE user's ranked list against a relevant set."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Hashable, Sequence, Set

from ..exceptions import InvalidParameterError
from ..models.base import validate_k


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _compute(self, ranking: Sequence[Hashable], relevant: Set[Hashable]) -> float: ...

    def compute(self, ranking: Sequence[Hashable], relevant: Set[Hashable]) -> float:
        if not relevant:
            raise InvalidParameterError("relevant set must not be empty")
        if len(ranking) == 0:
            return 0.0                    # nothing recommended -> nothing found
        return self._compute(ranking, set(relevant))


class _AtKMetric(BaseMetric):
    def __init__(self, k: int):
        self.k = validate_k(k)

    @property
    def name(self) -> str:
        return f"{type(self).__name__.replace('AtK', '')}@{self.k}"


class RecallAtK(_AtKMetric):
    def _compute(self, ranking, relevant):
        hits = len(set(ranking[:self.k]) & relevant)
        return hits / len(relevant)


class NDCGAtK(_AtKMetric):
    def _compute(self, ranking, relevant):
        dcg = sum(1.0 / math.log2(pos + 2)
                  for pos, item in enumerate(ranking[:self.k]) if item in relevant)
        ideal = sum(1.0 / math.log2(pos + 2) for pos in range(min(self.k, len(relevant))))
        return dcg / ideal


class AUC(BaseMetric):
    """Probability that a random relevant item is ranked above a random non-relevant one.
    Needs the FULL ranking of candidates. Returns NaN when undefined (no negatives)."""

    @property
    def name(self) -> str:
        return "AUC"

    def _compute(self, ranking, relevant):
        n = len(ranking)
        positions = [p for p, item in enumerate(ranking) if item in relevant]
        n_pos = len(positions)
        n_neg = n - n_pos
        if n_pos == 0:
            return 0.0
        if n_neg == 0:
            return float("nan")
        # negatives ranked below each positive
        below = sum((n - 1 - p) - (n_pos - 1 - r) for r, p in enumerate(positions))
        return below / (n_pos * n_neg)
