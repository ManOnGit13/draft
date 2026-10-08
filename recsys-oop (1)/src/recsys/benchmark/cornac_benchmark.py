"""Adapter that wraps a Cornac model as a BaseRecommender.

Cornac is only USED (imported lazily, never modified). Because the adapter
implements our interface, the SAME Evaluator, split and metrics are applied to
our models and to Cornac models, which makes the comparison fair.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np

from ..data.dataset import Dataset
from ..evaluation.evaluator import EvaluationResult, Evaluator
from ..exceptions import ConfigError
from ..models.base import BaseRecommender


def cornac_available() -> bool:
    try:
        import cornac  # noqa: F401
        return True
    except ImportError:
        return False


def _import_cornac():
    try:
        import cornac
        return cornac
    except ImportError as exc:
        raise ConfigError("cornac is not installed (pip install cornac)") from exc


class CornacRecommender(BaseRecommender):
    SUPPORTED = ("MF", "BPR")

    def __init__(self, cornac_name: str = "BPR", **params: Any):
        super().__init__()
        if cornac_name not in self.SUPPORTED:
            raise ConfigError(f"unsupported Cornac model {cornac_name!r}; use {self.SUPPORTED}")
        self.name = f"cornac-{cornac_name}"
        self._cornac_name, self._params = cornac_name, params
        self._cmodel = None
        self._train_set = None
        self._item_cols: np.ndarray = np.zeros(0, dtype=int)

    def _fit(self, dataset: Dataset) -> None:
        cornac = _import_cornac()
        from cornac.data import Dataset as CornacDataset
        pos = [x for x in dataset.interactions if x.rating >= dataset.rating_threshold]
        triples = [(x.user_id, x.item_id, 1.0) for x in pos]          # implicit feedback
        train_set = CornacDataset.from_uir(triples, seed=123)
        model = getattr(cornac.models, self._cornac_name)(**self._params)
        model.fit(train_set)
        self._cmodel, self._train_set = model, train_set
        # our item index -> cornac item index (or -1 if the item has no positive)
        self._item_cols = np.array(
            [train_set.iid_map.get(i, -1) for i in dataset.item_ids], dtype=int)
        self._user_ids = dataset.user_ids

    def _score_user(self, user_idx: int) -> np.ndarray:
        uid = self._train_set.uid_map.get(self._user_ids[user_idx])
        n_items = len(self._item_cols)
        if uid is None:
            return np.zeros(n_items)
        raw = np.asarray(self._cmodel.score(uid), dtype=np.float64)
        out = np.full(n_items, raw.min() - 1.0)
        valid = self._item_cols >= 0
        out[valid] = raw[self._item_cols[valid]]
        return out


class CornacBenchmark:
    """Runs Cornac models through our Evaluator on the given train/test split."""

    def __init__(self, evaluator: Evaluator, models: Dict[str, Dict[str, Any]] = None):
        self._evaluator = evaluator
        self._models = models or {"BPR": {"k": 32, "max_iter": 100, "learning_rate": 0.05,
                                          "lambda_reg": 0.01, "seed": 42},
                                  "MF": {"k": 32, "max_iter": 100, "learning_rate": 0.01,
                                         "lambda_reg": 0.02, "seed": 42}}

    def run(self, train: Dataset, test: Dataset) -> List[EvaluationResult]:
        _import_cornac()
        results = []
        for name, params in self._models.items():
            model = CornacRecommender(name, **params)
            t0 = time.perf_counter()
            model.fit(train)
            train_time = time.perf_counter() - t0
            res = self._evaluator.evaluate(model, test, label=f"cornac-{name}")
            res.seconds = train_time
            results.append(res)
        return results
