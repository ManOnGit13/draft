"""Immutable domain entities. Validation happens at construction time."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..exceptions import DataError


def _clean_id(value: Any, kind: str) -> str:
    if value is None:
        raise DataError(f"{kind} id must not be None")
    text = str(value).strip()
    if not text:
        raise DataError(f"{kind} id must not be empty")
    return text


@dataclass(frozen=True)
class User:
    user_id: str
    attributes: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "user_id", _clean_id(self.user_id, "user"))


@dataclass(frozen=True)
class Item:
    item_id: str
    title: str = ""
    genres: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "item_id", _clean_id(self.item_id, "item"))
        object.__setattr__(self, "genres", tuple(self.genres))


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float = 1.0
    timestamp: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "user_id", _clean_id(self.user_id, "user"))
        object.__setattr__(self, "item_id", _clean_id(self.item_id, "item"))
        try:
            rating = float(self.rating)
        except (TypeError, ValueError):
            raise DataError(f"rating must be numeric, got {self.rating!r}")
        if not math.isfinite(rating):
            raise DataError(f"rating must be finite, got {self.rating!r}")
        object.__setattr__(self, "rating", rating)
