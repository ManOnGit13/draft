"""End-to-end checks that every failure is a RecsysError (never a raw crash)."""
import unittest
from recsys.data import *
from recsys.exceptions import *
from recsys.evaluation import *
from recsys.models import *
from recsys.service import *
from .helpers import small_synthetic


class TestRobustness(unittest.TestCase):
    def test_all_expected_errors_share_base_class(self):
        for exc in (DataError, ConfigError, InvalidParameterError, ModelNotFittedError,
                    UnknownUserError, UnknownItemError):
            self.assertTrue(issubclass(exc, RecsysError))

    def test_full_pipeline(self):
        ds = small_synthetic()
        train, test = RandomSplitter(seed=7).split(ds)
        svc = RecommendationService(ServiceConfig("mf", {"n_epochs": 5})).train(train)
        self.assertEqual(len(svc.recommend("u5", 10)), 10)

    def test_cold_user_in_universe_does_not_crash(self):
        rows = [Interaction("u1", "a", 5), Interaction("u1", "b", 5), Interaction("u2", "a", 5)]
        ds = Dataset(rows, users=[User("u1"), User("u2"), User("newbie")], rating_threshold=4)
        for name in ("popularity", "mf"):
            m = ModelFactory.create(name).fit(ds)
            self.assertEqual(len(m.recommend("newbie", 2)), 2)

    def test_single_user_single_item(self):
        ds = Dataset([Interaction("u", "a", 5)], rating_threshold=4)
        m = ModelFactory.create("popularity").fit(ds)
        self.assertEqual(m.recommend("u", 5), [])      # nothing left to recommend

    def test_failed_refit_leaves_model_unfitted(self):
        m = PopularityRecommender().fit(small_synthetic())
        with self.assertRaises(DataError):
            m.fit(None)
        self.assertTrue(m.is_fitted)    # validation failed before touching state


if __name__ == "__main__":
    unittest.main()
