from recsys.data import Interaction, Dataset, SyntheticLoader


def tiny_dataset(threshold=4.0):
    rows = [("u1", "a", 5), ("u1", "b", 4), ("u1", "c", 2),
            ("u2", "a", 5), ("u2", "b", 5), ("u2", "d", 4),
            ("u3", "a", 4), ("u3", "d", 5)]
    return Dataset([Interaction(u, i, r, 100 + n) for n, (u, i, r) in enumerate(rows)],
                   rating_threshold=threshold)


def small_synthetic():
    return SyntheticLoader(n_users=60, n_items=50, n_genres=4,
                           min_per_user=8, max_per_user=20, seed=1).load()
