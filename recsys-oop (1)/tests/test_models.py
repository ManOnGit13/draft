import unittest
import numpy as np
from recsys.data import Interaction, Dataset
from recsys.exceptions import *
from recsys.models import *
from .helpers import tiny_dataset, small_synthetic


class ModelContract:
    """Behaviour every recommender must satisfy (polymorphism / Liskov)."""
    make = None

    def fitted(self):
        return self.make().fit(small_synthetic())

    def test_before_fit(self):
        m = self.make()
        self.assertFalse(m.is_fitted)
        for call in (lambda: m.recommend("u0"), lambda: m.rank("u0"),
                     lambda: m.score("u0"), lambda: m.dataset):
            with self.assertRaises(ModelNotFittedError):
                call()

    def test_recommend_excludes_seen_and_sorted(self):
        m = self.fitted()
        recs = m.recommend("u1", k=10)
        self.assertEqual(len(recs), 10)
        ids = [i for i, _ in recs]
        self.assertEqual(len(set(ids)), 10)
        seen = {m.dataset.item_id(int(j)) for j in m.dataset.seen_item_indices(m.dataset.user_index("u1"))}
        self.assertFalse(set(ids) & seen)
        scores = [s for _, s in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_include_seen(self):
        m = self.fitted()
        self.assertEqual(len(m.rank("u1", exclude_seen=False)), m.dataset.n_items)

    def test_k_larger_than_candidates(self):
        m = self.fitted()
        n_cand = m.dataset.n_items - len(m.dataset.seen_item_indices(m.dataset.user_index("u1")))
        self.assertEqual(len(m.recommend("u1", k=10_000)), n_cand)

    def test_invalid_k(self):
        m = self.fitted()
        for k in (0, -3, 2.5, "5", None, True):
            with self.subTest(k=k), self.assertRaises(InvalidParameterError):
                m.recommend("u1", k=k)

    def test_unknown_user_and_item(self):
        m = self.fitted()
        with self.assertRaises(UnknownUserError):
            m.recommend("ghost", k=5)
        with self.assertRaises(UnknownItemError):
            m.score("u1", ["m1", "nope"])

    def test_score_subset(self):
        m = self.fitted()
        full = m.score("u1")
        sub = m.score("u1", ["m3", "m7"])
        np.testing.assert_allclose(sub, full[[m.dataset.item_index("m3"), m.dataset.item_index("m7")]])

    def test_fit_invalid_data(self):
        with self.assertRaises(DataError):
            self.make().fit("not a dataset")
        no_pos = Dataset([Interaction("u", "a", 1), Interaction("u", "b", 2)], rating_threshold=4)
        with self.assertRaises(DataError):
            self.make().fit(no_pos)

    def test_scores_are_not_mutable_from_outside(self):
        m = self.fitted()
        s = m.score("u1"); s[:] = -999
        self.assertGreater(m.score("u1").max(), -999)


class TestPopularity(ModelContract, unittest.TestCase):
    make = staticmethod(lambda: PopularityRecommender())

    def test_ranks_by_count(self):
        m = PopularityRecommender().fit(tiny_dataset())
        # positives: a=3, b=2, d=2, c=0
        self.assertEqual(m.top_popular(1)[0], ("a", 3.0))
        self.assertEqual([i for i, _ in m.top_popular(4)][-1], "c")

    def test_same_ranking_for_all_users_when_include_seen(self):
        m = PopularityRecommender().fit(tiny_dataset())
        np.testing.assert_array_equal(m.rank("u1", False), m.rank("u2", False))


class TestMF(ModelContract, unittest.TestCase):
    make = staticmethod(lambda: MatrixFactorizationRecommender(n_factors=8, n_epochs=10))

    def test_loss_decreases(self):
        m = self.fitted()
        self.assertLess(m.loss_history_[-1], m.loss_history_[0])

    def test_reproducible(self):
        a, b = self.fitted(), self.fitted()
        np.testing.assert_allclose(a.score("u3"), b.score("u3"))

    def test_cold_user_uses_item_bias(self):
        ds = Dataset([Interaction("u1", "a", 5), Interaction("u1", "b", 5),
                      Interaction("u2", "a", 5), Interaction("cold", "c", 1)],
                     rating_threshold=4)
        m = MatrixFactorizationRecommender(n_factors=4, n_epochs=5).fit(ds)
        self.assertEqual(len(m.recommend("cold", 2)), 2)   # does not crash

    def test_invalid_hyperparameters(self):
        bad = [dict(n_factors=0), dict(n_factors=-1), dict(n_factors=2.5), dict(n_epochs=0),
               dict(batch_size=0), dict(learning_rate=0), dict(learning_rate=-0.1),
               dict(learning_rate="fast"), dict(reg=-1)]
        for params in bad:
            with self.subTest(params), self.assertRaises(ConfigError):
                MatrixFactorizationRecommender(**params)


class TestFactory(unittest.TestCase):
    def test_create(self):
        self.assertIsInstance(ModelFactory.create("popularity"), PopularityRecommender)
        self.assertIsInstance(ModelFactory.create("mf", n_factors=4), MatrixFactorizationRecommender)
        self.assertIn("mf", available_models())

    def test_invalid(self):
        for name in ("nope", None, 3):
            with self.assertRaises(ConfigError):
                ModelFactory.create(name)
        with self.assertRaises(ConfigError):
            ModelFactory.create("mf", not_a_param=1)
        with self.assertRaises(ConfigError):
            ModelFactory.create("popularity", n_factors=3)


if __name__ == "__main__":
    unittest.main()
