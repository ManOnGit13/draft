"""Facade: the only class an application needs. Hides models, datasets, indices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..data.dataset import Dataset
from ..data.entities import Interaction, User
from ..exceptions import DataError, InvalidParameterError, ModelNotFittedError, UnknownUserError
from ..models.base import validate_k
from ..models.popularity import PopularityRecommender
from ..models.registry import ModelFactory
from .config import ServiceConfig


@dataclass(frozen=True)
class Recommendation:
    item_id: str
    title: str
    score: float
    is_fallback: bool = False


class RecommendationService:
    def __init__(self, config: Optional[ServiceConfig] = None):
        self._config = config or ServiceConfig()
        self._model = self._new_model()          # also validates the config early
        self._popularity: Optional[PopularityRecommender] = None
        self._dataset: Optional[Dataset] = None

    def _new_model(self):
        return ModelFactory.create(self._config.model_name, **self._config.model_params)

    # ---- state ----
    @property
    def is_trained(self) -> bool:
        return self._dataset is not None and self._model.is_fitted

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise ModelNotFittedError("call train() before using the service")

    def has_user(self, user_id: str) -> bool:
        self._require_trained()
        return self._dataset.has_user(user_id)

    def user_ids(self) -> List[str]:
        self._require_trained()
        return self._dataset.user_ids

    # ---- training ----
    def train(self, dataset: Dataset) -> "RecommendationService":
        """Trains a fresh model and swaps it in only on success, so a failed
        train() never destroys a previously working service."""
        if not isinstance(dataset, Dataset):
            raise DataError("train() expects a Dataset")
        model = self._new_model().fit(dataset)
        popularity = model if isinstance(model, PopularityRecommender) \
            else PopularityRecommender().fit(dataset)
        self._model, self._popularity, self._dataset = model, popularity, dataset
        return self

    def add_user(self, user_id: str, ratings: Dict[str, float]) -> "RecommendationService":
        """Register a NEW user with their ratings ({item_id: rating}) and retrain."""
        self._require_trained()
        if not isinstance(user_id, str) or not user_id.strip():
            raise InvalidParameterError("user_id must be a non-empty string")
        user_id = user_id.strip()
        if self._dataset.has_user(user_id):
            raise DataError(f"user {user_id!r} already exists")
        if not isinstance(ratings, dict) or not ratings:
            raise InvalidParameterError("ratings must be a non-empty {item_id: rating} dict")
        for item_id in ratings:
            self._dataset.item_index(item_id)               # raises UnknownItemError
        ds = self._dataset
        new = [Interaction(user_id, i, r) for i, r in ratings.items()]
        updated = Dataset(list(ds.interactions) + new, list(ds.users) + [User(user_id)],
                          ds.items, ds.rating_threshold)
        return self.train(updated)

    # ---- queries ----
    def popular_items(self, k: Optional[int] = None) -> List[Recommendation]:
        self._require_trained()
        k = validate_k(self._config.default_k if k is None else k)
        return self._wrap(self._popularity.top_popular(k), fallback=False)

    def recommend(self, user_id: str, k: Optional[int] = None) -> List[Recommendation]:
        self._require_trained()
        k = validate_k(self._config.default_k if k is None else k)
        if self._dataset.has_user(user_id):
            pairs = self._model.recommend(user_id, k, self._config.exclude_seen)
            return self._wrap(pairs, fallback=False)
        if self._config.cold_start_fallback:
            return self._wrap(self._popularity.top_popular(k), fallback=True)
        raise UnknownUserError(f"unknown user: {user_id!r}")

    def _wrap(self, pairs, fallback: bool) -> List[Recommendation]:
        ds = self._dataset
        return [Recommendation(i, ds.get_item(i).title, s, fallback) for i, s in pairs]
