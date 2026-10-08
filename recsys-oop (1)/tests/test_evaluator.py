import unittest
from recsys.data import RandomSplitter, Interaction, Dataset
from recsys.evaluation import *
from recsys.exceptions import *
from recsys.models import *
from .helpers import small_synthetic


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.train, self.test = RandomSplitter(0.25, seed=0).split(small_synthetic())
        self.ev = Evaluator([RecallAtK(10), NDCGAtK(10), AUC()])

    def test_runs_and_in_range(self):
        for name in ("popularity", "mf"):
            m = ModelFactory.create(name).fit(self.train)
            res = self.ev.evaluate(m, self.test)
            self.assertEqual(set(res.scores), {"Recall@10", "NDCG@10", "AUC"})
            for v in res.scores.values():
                self.assertTrue(0.0 <= v <= 1.0)
            self.assertGreater(res.n_users_evaluated, 0)

    def test_mf_beats_popularity_on_structured_data(self):
        pop = self.ev.evaluate(PopularityRecommender().fit(self.train), self.test)
        mf = self.ev.evaluate(MatrixFactorizationRecommender(n_epochs=30).fit(self.train), self.test)
        self.assertGreater(mf.scores["NDCG@10"], pop.scores["NDCG@10"])

    def test_unfitted_model(self):
        with self.assertRaises(ModelNotFittedError):
            self.ev.evaluate(PopularityRecommender(), self.test)

    def test_bad_metric_lists(self):
        for bad in ([], [RecallAtK(5), RecallAtK(5)], ["recall"]):
            with self.assertRaises(InvalidParameterError):
                Evaluator(bad)

    def test_no_evaluable_users(self):
        m = PopularityRecommender().fit(self.train)
        ds = Dataset([Interaction("stranger", "m1", 5)])
        with self.assertRaises(InvalidParameterError):
            self.ev.evaluate(m, ds)

    def test_users_without_relevant_are_skipped(self):
        m = PopularityRecommender().fit(self.train)
        low = Dataset([Interaction("u0", "m1", 1), Interaction("u1", "m2", 5)],
                      users=self.train.users, items=self.train.items, rating_threshold=4)
        res = self.ev.evaluate(m, low)
        self.assertEqual((res.n_users_evaluated, res.n_users_skipped), (1, 1))


if __name__ == "__main__":
    unittest.main()
