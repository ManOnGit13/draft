from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict

from ..exceptions import ConfigError
from ..models.registry import available_models


@dataclass
class ServiceConfig:
    model_name: str = "mf"
    model_params: Dict[str, Any] = field(default_factory=dict)
    default_k: int = 10
    exclude_seen: bool = True
    cold_start_fallback: bool = True     # unknown user -> most popular items

    def __post_init__(self):
        if self.model_name not in available_models():
            raise ConfigError(f"unknown model {self.model_name!r}; available: {available_models()}")
        if not isinstance(self.model_params, dict):
            raise ConfigError("model_params must be a dict")
        if isinstance(self.default_k, bool) or not isinstance(self.default_k, int) or self.default_k <= 0:
            raise ConfigError(f"default_k must be a positive integer, got {self.default_k!r}")
        for flag in ("exclude_seen", "cold_start_fallback"):
            if not isinstance(getattr(self, flag), bool):
                raise ConfigError(f"{flag} must be a boolean")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceConfig":
        if not isinstance(data, dict):
            raise ConfigError("config must be a mapping")
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "ServiceConfig":
        try:
            with open(path, encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except FileNotFoundError:
            raise ConfigError(f"config file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file is not valid JSON: {exc}") from exc
