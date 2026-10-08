import os, tempfile, unittest
from recsys.data import *
from recsys.exceptions import *
from .helpers import tiny_dataset, small_synthetic


class TestEntities(unittest.TestCase):
    def test_normal(self):
        x = Interaction(1, 2, "4.5")
        self.assertEqual((x.user_id, x.item_id, x.rating), ("1", "2", 4.5))

    def test_invalid(self):
        for bad in (dict(user_id="", item_id="a"), dict(user_id=None, item_id="a"),
                    dict(user_id="u", item_id="a", rating="x"),
                    dict(user_id="u", item_id="a", rating=float("nan")),
                    dict(user_id="u", item_id="a", rating=float("inf"))):
            with self.assertRaises(DataError):
                Interaction(**bad)


class TestDataset(unittest.TestCase):
    def test_basic(self):
        ds = tiny_dataset()
        self.assertEqual((ds.n_users, ds.n_items, ds.n_interactions), (3, 4, 8))
        self.assertEqual(ds.positive_item_ids("u1"), {"a", "b"})   # c has rating 2
        self.assertEqual(len(ds.seen_item_indices(ds.user_index("u1"))), 3)

    def test_empty(self):
        with self.assertRaises(DataError):
            Dataset([])

    def test_duplicate(self):
        with self.assertRaises(DataError):
            Dataset([Interaction("u", "a", 5), Interaction("u", "a", 4)])

    def test_unknown_lookups(self):
        ds = tiny_dataset()
        with self.assertRaises(UnknownUserError):
            ds.user_index("nobody")
        with self.assertRaises(UnknownItemError):
            ds.item_index("zzz")
        self.assertFalse(ds.has_user(None))

    def test_wrong_type(self):
        with self.assertRaises(DataError):
            Dataset([("u", "a", 5)])

    def test_interaction_outside_universe(self):
        with self.assertRaises(DataError):
            Dataset([Interaction("u", "a")], users=[User("x")], items=[Item("a")])


class TestLoaders(unittest.TestCase):
    def _write(self, d, name, text):
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return p

    def test_csv_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write(d, "x.csv", "user_id,item_id,rating\nu1,a,5\nu2,b,3\n")
            self.assertEqual(CSVLoader(p).load().n_interactions, 2)

    def test_csv_problems(self):
        with tempfile.TemporaryDirectory() as d:
            cases = {"empty.csv": "", "header.csv": "user_id,item_id,rating\n",
                     "cols.csv": "a,b\n1,2\n", "nan.csv": "user_id,item_id,rating\nu,a,nan\n",
                     "blank.csv": "user_id,item_id,rating\nu,a,\n",
                     "text.csv": "user_id,item_id,rating\nu,a,great\n"}
            for name, text in cases.items():
                with self.subTest(name), self.assertRaises(DataError):
                    CSVLoader(self._write(d, name, text)).load()

    def test_missing_file(self):
        with self.assertRaises(DataError):
            CSVLoader("/no/such/file.csv").load()
        with self.assertRaises(DataError):
            MovieLensLoader("/no/such/dir").load()

    def test_movielens_100k_format(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "u.data", "1\t10\t5\t881250949\n1\t20\t3\t881250950\n2\t10\t4\t881250951\n")
            self._write(d, "u.item", "10|Toy Story (1995)|01-Jan-1995||url|0|0|0|1|1|1" + "|0" * 13 + "\n"
                                      "20|GoldenEye (1995)|01-Jan-1995||url" + "|0" * 19 + "\n")
            ds = MovieLensLoader(d).load()
            self.assertEqual(ds.get_item("10").title, "Toy Story (1995)")
            self.assertEqual(ds.get_item("10").genres, ("Animation", "Children's", "Comedy"))
            self.assertEqual(ds.positive_item_ids("1"), {"10"})

    def test_movielens_bad_variant_and_line(self):
        from recsys.exceptions import InvalidParameterError
        with self.assertRaises(InvalidParameterError):
            MovieLensLoader(".", "10m")
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "u.data", "1\t10\n")
            with self.assertRaises(DataError):
                MovieLensLoader(d).load()

    def test_synthetic_reproducible(self):
        a, b = small_synthetic(), small_synthetic()
        self.assertEqual(a.interactions, b.interactions)


class TestSplitters(unittest.TestCase):
    def test_random_split_disjoint(self):
        ds = small_synthetic()
        tr, te = RandomSplitter(0.2, seed=1).split(ds)
        a = {(x.user_id, x.item_id) for x in tr.interactions}
        b = {(x.user_id, x.item_id) for x in te.interactions}
        self.assertFalse(a & b)
        self.assertEqual(len(a) + len(b), ds.n_interactions)
        self.assertEqual(tr.n_users, te.n_users)

    def test_deterministic(self):
        ds = small_synthetic()
        t1 = RandomSplitter(seed=3).split(ds)[1].interactions
        t2 = RandomSplitter(seed=3).split(ds)[1].interactions
        self.assertEqual(t1, t2)

    def test_leave_last_out(self):
        tr, te = LeaveLastOutSplitter().split(tiny_dataset())
        self.assertEqual(te.n_interactions, 3)
        self.assertEqual(te.positive_item_ids("u3"), {"d"})

    def test_invalid(self):
        from recsys.exceptions import InvalidParameterError
        for r in (0, 1, -0.1, 1.5, "0.2", True):
            with self.assertRaises(InvalidParameterError):
                RandomSplitter(r)
        no_ts = Dataset([Interaction("u", "a", 5), Interaction("u", "b", 5)])
        with self.assertRaises(DataError):
            LeaveLastOutSplitter().split(no_ts)

    def test_too_small(self):
        one_each = Dataset([Interaction("u1", "a", 5), Interaction("u2", "b", 5)])
        with self.assertRaises(DataError):
            RandomSplitter().split(one_each)


if __name__ == "__main__":
    unittest.main()
