import math, unittest
from recsys.evaluation import *
from recsys.exceptions import InvalidParameterError


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(RecallAtK(3).compute([1, 2, 3, 4], {2, 4}), 0.5)
        self.assertEqual(RecallAtK(10).compute([1, 2], {1, 2}), 1.0)
        self.assertEqual(RecallAtK(1).compute([9, 1], {1}), 0.0)

    def test_ndcg_perfect_and_worst(self):
        self.assertAlmostEqual(NDCGAtK(3).compute([1, 2, 3], {1, 2}), 1.0)
        self.assertEqual(NDCGAtK(2).compute([5, 6, 1], {1}), 0.0)

    def test_ndcg_value(self):
        # single relevant item at rank 2 -> (1/log2(3)) / 1
        self.assertAlmostEqual(NDCGAtK(5).compute([9, 1, 8], {1}), 1 / math.log2(3))

    def test_auc(self):
        self.assertEqual(AUC().compute([1, 2, 3, 4], {1, 2}), 1.0)
        self.assertEqual(AUC().compute([3, 4, 1, 2], {1, 2}), 0.0)
        self.assertAlmostEqual(AUC().compute([1, 3, 2, 4], {1, 2}), 0.75)
        self.assertTrue(math.isnan(AUC().compute([1, 2], {1, 2})))   # no negatives

    def test_empty_inputs(self):
        for metric in (RecallAtK(5), NDCGAtK(5), AUC()):
            self.assertEqual(metric.compute([], {1}), 0.0)           # empty ranking
            with self.assertRaises(InvalidParameterError):           # empty relevant set
                metric.compute([1, 2], set())

    def test_invalid_k_and_names(self):
        for k in (0, -1, 1.5, "3"):
            with self.assertRaises(InvalidParameterError):
                RecallAtK(k)
        self.assertEqual(NDCGAtK(10).name, "NDCG@10")
        self.assertEqual(RecallAtK(5).name, "Recall@5")
        self.assertEqual(AUC().name, "AUC")

    def test_bounds(self):
        import random
        rnd = random.Random(0)
        for _ in range(200):
            ranking = rnd.sample(range(30), 30)
            rel = set(rnd.sample(range(30), rnd.randint(1, 29)))
            for m in (RecallAtK(5), NDCGAtK(5), AUC()):
                self.assertTrue(0.0 <= m.compute(ranking, rel) <= 1.0)


if __name__ == "__main__":
    unittest.main()
