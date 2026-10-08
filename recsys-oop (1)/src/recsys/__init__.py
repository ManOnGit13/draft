"""recsys - a small object-oriented recommendation system."""
from .exceptions import (
    RecsysError, DataError, ConfigError, InvalidParameterError,
    ModelNotFittedError, UnknownUserError, UnknownItemError,
)

__version__ = "1.0.0"
__all__ = [
    "RecsysError", "DataError", "ConfigError", "InvalidParameterError",
    "ModelNotFittedError", "UnknownUserError", "UnknownItemError",
]
