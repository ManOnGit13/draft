"""Loaders turn an external source into a Dataset (Template Method + Strategy)."""
from __future__ import annotations

import csv
import math
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..exceptions import DataError, InvalidParameterError
from .dataset import Dataset
from .entities import Interaction, Item


class BaseLoader(ABC):
    """Abstract loader. Subclasses implement _read(); load() adds validation."""

    def __init__(self, rating_threshold: float = 4.0):
        self.rating_threshold = rating_threshold

    @abstractmethod
    def _read(self) -> Tuple[List[Interaction], Optional[List[Item]]]:
        """Return (interactions, items-or-None)."""

    def load(self) -> Dataset:
        interactions, items = self._read()
        if not interactions:
            raise DataError(f"{type(self).__name__}: source contains no interactions")
        return Dataset(interactions, items=items, rating_threshold=self.rating_threshold)


def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise DataError(f"file not found: {path}")


class MovieLensLoader(BaseLoader):
    """Reads MovieLens 100K (u.data/u.item) or 1M (ratings.dat/movies.dat)."""

    GENRES_100K = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
                   "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                   "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

    def __init__(self, directory: str, variant: str = "100k", rating_threshold: float = 4.0):
        super().__init__(rating_threshold)
        if variant not in ("100k", "1m"):
            raise InvalidParameterError("variant must be '100k' or '1m'")
        self.directory = directory
        self.variant = variant

    def _read(self):
        if self.variant == "100k":
            return self._read_100k()
        return self._read_1m()

    def _read_ratings(self, path: str, sep: Optional[str]) -> List[Interaction]:
        _require_file(path)
        out = []
        with open(path, encoding="latin-1") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                parts = line.strip().split(sep) if sep else line.split()
                if len(parts) < 3:
                    raise DataError(f"{path}:{n}: expected >=3 fields, got {len(parts)}")
                ts = int(parts[3]) if len(parts) > 3 and parts[3].lstrip("-").isdigit() else None
                out.append(Interaction(parts[0], parts[1], parts[2], ts))
        return out

    def _read_100k(self):
        inter = self._read_ratings(os.path.join(self.directory, "u.data"), "\t")
        items = None
        path = os.path.join(self.directory, "u.item")
        if os.path.isfile(path):
            items = []
            with open(path, encoding="latin-1") as fh:
                for line in fh:
                    f = line.rstrip("\n").split("|")
                    if len(f) < 5:
                        continue
                    flags = f[5:]
                    genres = tuple(g for g, v in zip(self.GENRES_100K, flags) if v == "1")
                    items.append(Item(f[0], f[1], genres))
            items = self._complete_items(inter, items)
        return inter, items

    def _read_1m(self):
        inter = self._read_ratings(os.path.join(self.directory, "ratings.dat"), "::")
        items = None
        path = os.path.join(self.directory, "movies.dat")
        if os.path.isfile(path):
            items = []
            with open(path, encoding="latin-1") as fh:
                for line in fh:
                    f = line.rstrip("\n").split("::")
                    if len(f) >= 3:
                        items.append(Item(f[0], f[1], tuple(f[2].split("|"))))
            items = self._complete_items(inter, items)
        return inter, items

    @staticmethod
    def _complete_items(inter, items):
        """Make sure every rated item exists in the catalogue."""
        known = {i.item_id for i in items}
        missing = sorted({x.item_id for x in inter} - known)
        return items + [Item(m) for m in missing]


class CSVLoader(BaseLoader):
    """Generic CSV loader with a header row."""

    def __init__(self, path: str, user_col: str = "user_id", item_col: str = "item_id",
                 rating_col: Optional[str] = "rating", timestamp_col: Optional[str] = None,
                 delimiter: str = ",", rating_threshold: float = 4.0):
        super().__init__(rating_threshold)
        self.path, self.user_col, self.item_col = path, user_col, item_col
        self.rating_col, self.timestamp_col, self.delimiter = rating_col, timestamp_col, delimiter

    def _read(self):
        _require_file(self.path)
        out: List[Interaction] = []
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=self.delimiter)
            if reader.fieldnames is None:
                raise DataError(f"{self.path}: file is empty")
            needed = [c for c in (self.user_col, self.item_col, self.rating_col,
                                  self.timestamp_col) if c]
            missing = [c for c in needed if c not in reader.fieldnames]
            if missing:
                raise DataError(f"{self.path}: missing column(s) {missing}")
            for n, row in enumerate(reader, 2):
                rating = row[self.rating_col] if self.rating_col else 1.0
                if rating is None or str(rating).strip() == "":
                    raise DataError(f"{self.path}:{n}: missing rating")
                ts = None
                if self.timestamp_col and row[self.timestamp_col]:
                    try:
                        ts = int(float(row[self.timestamp_col]))
                    except ValueError:
                        raise DataError(f"{self.path}:{n}: bad timestamp") from None
                out.append(Interaction(row[self.user_col], row[self.item_col], rating, ts))
        return out, None


class SyntheticLoader(BaseLoader):
    """Generates a reproducible fake 'movie' dataset with genre-based taste.

    Used for offline demos and tests; not a substitute for a real dataset.
    """

    def __init__(self, n_users: int = 600, n_items: int = 400, n_genres: int = 8,
                 min_per_user: int = 20, max_per_user: int = 60, seed: int = 0,
                 rating_threshold: float = 4.0):
        super().__init__(rating_threshold)
        if min(n_users, n_items, n_genres) <= 0 or not 0 < min_per_user <= max_per_user:
            raise InvalidParameterError("synthetic sizes must be positive and min<=max")
        if max_per_user > n_items:
            raise InvalidParameterError("max_per_user cannot exceed n_items")
        self.n_users, self.n_items, self.n_genres = n_users, n_items, n_genres
        self.min_per_user, self.max_per_user, self.seed = min_per_user, max_per_user, seed

    def _read(self):
        rng = np.random.default_rng(self.seed)
        genre_of = rng.integers(0, self.n_genres, size=self.n_items)
        pop = 1.0 / np.arange(1, self.n_items + 1) ** 0.7       # Zipf-like popularity
        rng.shuffle(pop)
        items = [Item(f"m{j}", f"Movie {j}", (f"genre{genre_of[j]}",)) for j in range(self.n_items)]
        inter: List[Interaction] = []
        for u in range(self.n_users):
            liked = rng.choice(self.n_genres, size=min(2, self.n_genres), replace=False)
            fav = np.isin(genre_of, liked)
            w = pop * np.where(fav, 6.0, 1.0)
            n = int(rng.integers(self.min_per_user, self.max_per_user + 1))
            chosen = rng.choice(self.n_items, size=n, replace=False, p=w / w.sum())
            order = rng.permutation(n)
            for rank, j in enumerate(chosen[order]):
                if fav[j]:
                    r = rng.choice([4, 5], p=[0.5, 0.5])
                else:
                    r = rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.4, 0.1])
                inter.append(Interaction(f"u{u}", f"m{j}", float(r), 1_000_000 + rank))
        return inter, items
