"""Model registry + factory (Factory pattern)."""
from __future__ import annotations

from typing import Callable, Dict, List, Type

from ..exceptions import ConfigError

_REGISTRY: Dict[str, type] = {}


def register_model(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls
    return deco


def available_models() -> List[str]:
    return sorted(_REGISTRY)


class ModelFactory:
    @staticmethod
    def create(name: str, **params):
        if not isinstance(name, str) or name not in _REGISTRY:
            raise ConfigError(f"unknown model {name!r}; available: {available_models()}")
        try:
            return _REGISTRY[name](**params)
        except TypeError as exc:        # unexpected / missing hyper-parameter
            raise ConfigError(f"invalid parameters for model {name!r}: {exc}") from exc
