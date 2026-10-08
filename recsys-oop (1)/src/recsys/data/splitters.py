"""Splitters create (train, test) datasets that share one user/item universe."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..exceptions import DataError, InvalidParameterError
from .dataset import Dataset
from .entities import Interaction


class BaseSplitter(ABC):
    @abstractmethod
    def _pick_test(self, by_user: Dict[str, List[Interaction]]) -> List[Interaction]:
        """Return the interactions that go to the test set."""

    def split(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        by_user: Dict[str, List[Interaction]] = defaultdict(list)
        for x in dataset.interactions:
            by_user[x.user_id].append(x)
        test = self._pick_test(by_user)
        if not test:
            raise DataError("dataset too small to split: no user has enough interactions")
        test_keys = {(x.user_id, x.item_id) for x in test}
        train = [x for x in dataset.interactions if (x.user_id, x.item_id) not in test_keys]
        if not train:
            raise DataError("split left the training set empty")
        return dataset.with_interactions(train), dataset.with_interactions(test)


class RandomSplitter(BaseSplitter):
    """Per-user random hold-out of `test_ratio` of interactions (at least 1)."""

    def __init__(self, test_ratio: float = 0.2, seed: int = 42):
        if isinstance(test_ratio, bool) or not isinstance(test_ratio, (int, float)) \
                or not 0.0 < test_ratio < 1.0:
            raise InvalidParameterError(f"test_ratio must be in (0, 1), got {test_ratio!r}")
        self.test_ratio, self.seed = float(test_ratio), seed

    def _pick_test(self, by_user):
        rng = np.random.default_rng(self.seed)
        test: List[Interaction] = []
        for uid in sorted(by_user):
            rows = by_user[uid]
            if len(rows) < 2:
                continue                      # keep single interactions in train
            n_test = min(max(1, int(round(len(rows) * self.test_ratio))), len(rows) - 1)
            for j in rng.choice(len(rows), size=n_test, replace=False):
                test.append(rows[int(j)])
        return test


class LeaveLastOutSplitter(BaseSplitter):
    """Most recent interaction of each user goes to test (needs timestamps)."""

    def _pick_test(self, by_user):
        test: List[Interaction] = []
        for uid in sorted(by_user):
            rows = by_user[uid]
            if len(rows) < 2:
                continue
            if any(r.timestamp is None for r in rows):
                raise DataError("LeaveLastOutSplitter requires timestamps on all interactions")
            test.append(max(rows, key=lambda r: r.timestamp))
        return test
