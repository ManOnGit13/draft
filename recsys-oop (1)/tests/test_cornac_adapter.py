"""Tests the Cornac adapter against a STUB 'cornac' module (the real library is not
needed here). Real-Cornac results come from scripts/run_experiment.py."""
import sys, types, unittest
import numpy as np
from recsys.benchmark import CornacRecommender, CornacBenchmark
from recsys.evaluation import *
from recsys.exceptions import ConfigError
from recsys.data import RandomSplitter
from .helpers import small_synthetic


def install_stub():
    cornac = types.ModuleType("cornac")
    data = types.ModuleType("cornac.data")
    models = types.ModuleType("cornac.models")

    class FakeTrainSet:
        def __init__(self, triples):
            self.uid_map = {u: n for n, u in enumerate(dict.fromkeys(t[0] for t in triples))}
            self.iid_map = {i: n for n, i in enumerate(dict.fromkeys(t[1] for t in triples))}
            self.counts = np.zeros(len(self.iid_map))
            for _, i, _ in triples:
                self.counts[self.iid_map[i]] += 1

    class FakeDataset:
        @staticmethod
        def from_uir(triples, seed=None):
            return FakeTrainSet(list(triples))

    class FakeBPR:
        def __init__(self, **params): self.params = params
        def fit(self, ts): self.ts = ts
        def score(self, uid): return self.ts.counts.copy()      # popularity-like

    data.Dataset = FakeDataset
    models.BPR = models.MF = FakeBPR
    cornac.data, cornac.models = data, models
    sys.modules.update({"cornac": cornac, "cornac.data": data, "cornac.models": models})


class TestAdapter(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("cornac", "cornac.data", "cornac.models")}
        install_stub()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_adapter_works_with_our_evaluator(self):
        train, test = RandomSplitter(seed=0).split(small_synthetic())
        model = CornacRecommender("BPR", k=4).fit(train)
        self.assertEqual(len(model.recommend("u1", 5)), 5)
        res = Evaluator([RecallAtK(10), NDCGAtK(10)]).evaluate(model, test)
        self.assertEqual(res.model, "cornac-BPR")

    def test_benchmark_runs_all_models(self):
        train, test = RandomSplitter(seed=0).split(small_synthetic())
        out = CornacBenchmark(Evaluator([RecallAtK(10)])).run(train, test)
        self.assertEqual([r.model for r in out], ["cornac-BPR", "cornac-MF"])

    def test_unsupported_model(self):
        with self.assertRaises(ConfigError):
            CornacRecommender("NotAModel")


class TestWithoutCornac(unittest.TestCase):
    def test_clear_error_when_cornac_missing(self):
        saved = {k: sys.modules.pop(k, None) for k in ("cornac", "cornac.data", "cornac.models")}
        real = None
        try:
            import importlib.util
            real = importlib.util.find_spec("cornac")
            if real is not None:
                self.skipTest("real cornac installed")
            train, _ = RandomSplitter(seed=0).split(small_synthetic())
            with self.assertRaises(ConfigError):
                CornacRecommender("BPR").fit(train)
        finally:
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
