"""Bidirectional mapping between external ids (str) and dense indices."""
from __future__ import annotations

from typing import Iterable, List, Type

from ..exceptions import DataError, RecsysError


class IdMapper:
    def __init__(self, ids: Iterable[str], error_cls: Type[RecsysError] = DataError,
                 kind: str = "id"):
        self._ids: List[str] = list(ids)
        self._index = {raw: idx for idx, raw in enumerate(self._ids)}
        if len(self._index) != len(self._ids):
            raise DataError(f"duplicate {kind} ids are not allowed")
        self._error_cls = error_cls
        self._kind = kind

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, raw_id) -> bool:
        try:
            return raw_id in self._index
        except TypeError:
            return False

    def to_index(self, raw_id: str) -> int:
        try:
            return self._index[raw_id]
        except (KeyError, TypeError):
            raise self._error_cls(f"unknown {self._kind}: {raw_id!r}") from None

    def to_id(self, index: int) -> str:
        if not 0 <= index < len(self._ids):
            raise self._error_cls(f"{self._kind} index out of range: {index}")
        return self._ids[index]

    @property
    def ids(self) -> List[str]:
        return list(self._ids)
