from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from ..data.dataset import Dataset
from ..exceptions import InvalidParameterError, ModelNotFittedError, UnknownUserError
from ..models.base import BaseRecommender
from .metrics import BaseMetric


@dataclass
class EvaluationResult:
    model: str
    scores: Dict[str, float]
    n_users_evaluated: int
    n_users_skipped: int
    seconds: float = 0.0


class Evaluator:
    """Evaluates any BaseRecommender with any list of BaseMetric (composition)."""

    def __init__(self, metrics: Sequence[BaseMetric]):
        metrics = list(metrics)
        if not metrics:
            raise InvalidParameterError("Evaluator needs at least one metric")
        if not all(isinstance(m, BaseMetric) for m in metrics):
            raise InvalidParameterError("metrics must be BaseMetric instances")
        names = [m.name for m in metrics]
        if len(set(names)) != len(names):
            raise InvalidParameterError(f"duplicate metrics: {names}")
        self._metrics = metrics

    def evaluate(self, model: BaseRecommender, test: Dataset,
                 label: str = "") -> EvaluationResult:
        if not model.is_fitted:
            raise ModelNotFittedError("cannot evaluate a model that has not been fitted")
        train = model.dataset
        start = time.perf_counter()
        sums = {m.name: [] for m in self._metrics}
        skipped = 0
        for uid in test.users_with_interactions():
            relevant_ids = test.positive_item_ids(uid)
            if not relevant_ids or not train.has_user(uid):
                skipped += 1
                continue
            try:
                relevant = {train.item_index(i) for i in relevant_ids if train.has_item(i)}
                if not relevant:
                    skipped += 1
                    continue
                ranking = model.rank(uid).tolist()
            except UnknownUserError:
                skipped += 1
                continue
            for m in self._metrics:
                sums[m.name].append(m.compute(ranking, relevant))
        n_eval = max((len(v) for v in sums.values()), default=0)
        if n_eval == 0:
            raise InvalidParameterError("no evaluable users in the test set")
        scores = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in sums.items()}
        return EvaluationResult(label or model.name, scores, n_eval, skipped,
                                time.perf_counter() - start)
