"""Dataset: owns users, items and interactions; hides the index/sparse details."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import sparse

from ..exceptions import DataError, UnknownItemError, UnknownUserError
from .entities import Interaction, Item, User
from .id_mapper import IdMapper


class Dataset:
    """Immutable collection of interactions over a fixed user/item universe.

    rating_threshold: interactions with rating >= threshold are "positive"
    (implicit feedback); all interactions count as "seen".
    """

    def __init__(self, interactions: Iterable[Interaction],
                 users: Optional[Sequence[User]] = None,
                 items: Optional[Sequence[Item]] = None,
                 rating_threshold: float = 0.0):
        interactions = list(interactions)
        if not interactions:
            raise DataError("dataset has no interactions")
        for x in interactions:
            if not isinstance(x, Interaction):
                raise DataError(f"expected Interaction, got {type(x).__name__}")

        if users is None:
            users = [User(u) for u in sorted({x.user_id for x in interactions})]
        if items is None:
            items = [Item(i) for i in sorted({x.item_id for x in interactions})]
        self._users: Tuple[User, ...] = tuple(users)
        self._items: Tuple[Item, ...] = tuple(items)
        self._user_map = IdMapper([u.user_id for u in self._users], UnknownUserError, "user")
        self._item_map = IdMapper([i.item_id for i in self._items], UnknownItemError, "item")

        seen_pairs: Set[Tuple[str, str]] = set()
        u_idx, i_idx, ratings, stamps = [], [], [], []
        for x in interactions:
            if x.user_id not in self._user_map or x.item_id not in self._item_map:
                raise DataError(f"interaction refers to unknown user/item: {x.user_id}, {x.item_id}")
            key = (x.user_id, x.item_id)
            if key in seen_pairs:
                raise DataError(f"duplicate interaction for {key}")
            seen_pairs.add(key)
            u_idx.append(self._user_map.to_index(x.user_id))
            i_idx.append(self._item_map.to_index(x.item_id))
            ratings.append(x.rating)
            stamps.append(np.nan if x.timestamp is None else float(x.timestamp))

        self._interactions: Tuple[Interaction, ...] = tuple(interactions)
        self._u = np.asarray(u_idx, dtype=np.int64)
        self._i = np.asarray(i_idx, dtype=np.int64)
        self._r = np.asarray(ratings, dtype=np.float64)
        self._t = np.asarray(stamps, dtype=np.float64)
        self._rating_threshold = float(rating_threshold)
        self._seen_csr: Optional[sparse.csr_matrix] = None
        self._pos_csr: Optional[sparse.csr_matrix] = None

    # ---- sizes / ids ----
    @property
    def n_users(self) -> int:
        return len(self._user_map)

    @property
    def n_items(self) -> int:
        return len(self._item_map)

    @property
    def n_interactions(self) -> int:
        return len(self._interactions)

    @property
    def rating_threshold(self) -> float:
        return self._rating_threshold

    @property
    def users(self) -> Tuple[User, ...]:
        return self._users

    @property
    def items(self) -> Tuple[Item, ...]:
        return self._items

    @property
    def interactions(self) -> Tuple[Interaction, ...]:
        return self._interactions

    @property
    def user_ids(self) -> List[str]:
        return self._user_map.ids

    @property
    def item_ids(self) -> List[str]:
        return self._item_map.ids

    def has_user(self, user_id: str) -> bool:
        return user_id in self._user_map

    def has_item(self, item_id: str) -> bool:
        return item_id in self._item_map

    def user_index(self, user_id: str) -> int:
        return self._user_map.to_index(user_id)

    def item_index(self, item_id: str) -> int:
        return self._item_map.to_index(item_id)

    def user_id(self, index: int) -> str:
        return self._user_map.to_id(index)

    def item_id(self, index: int) -> str:
        return self._item_map.to_id(index)

    def get_item(self, item_id: str) -> Item:
        return self._items[self.item_index(item_id)]

    # ---- matrices / queries ----
    def _build(self, mask: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (np.ones(int(mask.sum()), dtype=np.float64), (self._u[mask], self._i[mask])),
            shape=(self.n_users, self.n_items))

    @property
    def seen_matrix(self) -> sparse.csr_matrix:
        if self._seen_csr is None:
            self._seen_csr = self._build(np.ones(len(self._u), dtype=bool))
        return self._seen_csr

    @property
    def positive_matrix(self) -> sparse.csr_matrix:
        if self._pos_csr is None:
            self._pos_csr = self._build(self._r >= self._rating_threshold)
        return self._pos_csr

    def seen_item_indices(self, user_idx: int) -> np.ndarray:
        m = self.seen_matrix
        return m.indices[m.indptr[user_idx]:m.indptr[user_idx + 1]]

    def positive_item_ids(self, user_id: str) -> Set[str]:
        u = self.user_index(user_id)
        m = self.positive_matrix
        return {self.item_id(int(j)) for j in m.indices[m.indptr[u]:m.indptr[u + 1]]}

    def users_with_interactions(self) -> List[str]:
        counts = np.diff(self.seen_matrix.indptr)
        return [self.user_id(int(u)) for u in np.nonzero(counts)[0]]

    def users_with_positives(self) -> List[str]:
        counts = np.diff(self.positive_matrix.indptr)
        return [self.user_id(int(u)) for u in np.nonzero(counts)[0]]

    def with_interactions(self, interactions: Iterable[Interaction]) -> "Dataset":
        """New dataset over the SAME user/item universe (used by splitters)."""
        return Dataset(interactions, self._users, self._items, self._rating_threshold)

    def summary(self) -> Dict[str, float]:
        density = self.n_interactions / (self.n_users * self.n_items)
        return {"users": self.n_users, "items": self.n_items,
                "interactions": self.n_interactions, "density": density}
