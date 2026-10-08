import json, os, tempfile, unittest
from recsys.exceptions import *
from recsys.models import PopularityRecommender
from recsys.service import *
from .helpers import small_synthetic


class TestService(unittest.TestCase):
    def setUp(self):
        self.ds = small_synthetic()
        self.svc = RecommendationService(
            ServiceConfig("mf", {"n_factors": 4, "n_epochs": 5}, default_k=7)).train(self.ds)

    def test_default_and_explicit_k(self):
        self.assertEqual(len(self.svc.recommend("u1")), 7)
        recs = self.svc.recommend("u1", k=3)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all(isinstance(r, Recommendation) and r.title for r in recs))
        self.assertFalse(any(r.is_fallback for r in recs))

    def test_before_train(self):
        svc = RecommendationService()
        self.assertFalse(svc.is_trained)
        with self.assertRaises(ModelNotFittedError):
            svc.recommend("u1")

    def test_unknown_user_fallback(self):
        recs = self.svc.recommend("ghost", k=4)
        self.assertEqual(len(recs), 4)
        self.assertTrue(all(r.is_fallback for r in recs))

    def test_unknown_user_without_fallback(self):
        svc = RecommendationService(ServiceConfig("popularity", cold_start_fallback=False)).train(self.ds)
        with self.assertRaises(UnknownUserError):
            svc.recommend("ghost")

    def test_invalid_k(self):
        for k in (0, -1, 3.2, "x"):
            with self.assertRaises(InvalidParameterError):
                self.svc.recommend("u1", k=k)
        with self.assertRaises(InvalidParameterError):          # also for unknown users
            self.svc.recommend("ghost", k=0)

    def test_train_bad_input_leaves_service_untrained(self):
        svc = RecommendationService()
        with self.assertRaises(DataError):
            svc.train([1, 2, 3])
        self.assertFalse(svc.is_trained)

    def test_popularity_service_reuses_model_as_fallback(self):
        svc = RecommendationService(ServiceConfig("popularity")).train(self.ds)
        self.assertEqual(len(svc.recommend("zzz", 2)), 2)


class TestAddUser(unittest.TestCase):
    def setUp(self):
        self.svc = RecommendationService(
            ServiceConfig("mf", {"n_factors": 4, "n_epochs": 5})).train(small_synthetic())

    def test_new_user_gets_personalised_results(self):
        self.assertFalse(self.svc.has_user("alice"))
        self.svc.add_user("alice", {"m1": 5, "m2": 4, "m3": 1})
        self.assertTrue(self.svc.has_user("alice"))
        recs = self.svc.recommend("alice", 5)
        self.assertEqual(len(recs), 5)
        self.assertFalse(any(r.is_fallback for r in recs))
        self.assertFalse({"m1", "m2", "m3"} & {r.item_id for r in recs})   # already seen

    def test_invalid_input(self):
        with self.assertRaises(UnknownItemError):
            self.svc.add_user("bob", {"nope": 5})
        with self.assertRaises(DataError):
            self.svc.add_user("u1", {"m1": 5})              # already exists
        for bad in ({}, None, [("m1", 5)]):
            with self.assertRaises(InvalidParameterError):
                self.svc.add_user("bob", bad)
        for bad_id in ("", "  ", None, 5):
            with self.assertRaises(InvalidParameterError):
                self.svc.add_user(bad_id, {"m1": 5})
        with self.assertRaises(DataError):
            self.svc.add_user("bob", {"m1": float("nan")})

    def test_failed_add_keeps_service_working(self):
        with self.assertRaises(RecsysError):
            self.svc.add_user("bob", {"m1": float("nan")})
        self.assertTrue(self.svc.is_trained)
        self.assertEqual(len(self.svc.recommend("u1", 3)), 3)
        self.assertFalse(self.svc.has_user("bob"))

    def test_before_train(self):
        svc = RecommendationService()
        for call in (lambda: svc.add_user("a", {"m1": 5}), lambda: svc.popular_items(),
                     lambda: svc.has_user("a"), lambda: svc.user_ids()):
            with self.assertRaises(ModelNotFittedError):
                call()

    def test_popular_items(self):
        self.assertEqual(len(self.svc.popular_items(4)), 4)
        with self.assertRaises(InvalidParameterError):
            self.svc.popular_items(0)


class TestConfig(unittest.TestCase):
    def test_invalid(self):
        bad = [dict(model_name="xyz"), dict(default_k=0), dict(default_k=-2), dict(default_k=2.5),
               dict(default_k=True), dict(model_params=[1]), dict(exclude_seen="yes"),
               dict(cold_start_fallback=1)]
        for kw in bad:
            with self.subTest(kw), self.assertRaises(ConfigError):
                ServiceConfig(**kw)

    def test_invalid_hyperparameters_surface_as_config_error(self):
        with self.assertRaises(ConfigError):
            RecommendationService(ServiceConfig("mf", {"n_factors": -5}))
        with self.assertRaises(ConfigError):
            RecommendationService(ServiceConfig("mf", {"bogus": 1}))

    def test_from_dict_and_json(self):
        with self.assertRaises(ConfigError):
            ServiceConfig.from_dict({"model_name": "mf", "typo_key": 1})
        with self.assertRaises(ConfigError):
            ServiceConfig.from_dict("nope")
        with tempfile.TemporaryDirectory() as d:
            good, broken = os.path.join(d, "g.json"), os.path.join(d, "b.json")
            with open(good, "w") as fh:
                json.dump({"model_name": "popularity", "default_k": 5}, fh)
            with open(broken, "w") as fh:
                fh.write("{not json")
            self.assertEqual(ServiceConfig.from_json(good).default_k, 5)
            for p in (broken, os.path.join(d, "missing.json")):
                with self.assertRaises(ConfigError):
                    ServiceConfig.from_json(p)

    def test_shipped_default_config_is_valid(self):
        path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.json")
        RecommendationService(ServiceConfig.from_json(path))


if __name__ == "__main__":
    unittest.main()
