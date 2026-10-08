"""Project-specific exception hierarchy.

Every error raised on purpose by this package derives from RecsysError, so a
caller can catch RecsysError to handle all expected failures in one place.
"""


class RecsysError(Exception):
    """Base class of all recsys errors."""


class DataError(RecsysError):
    """Data is empty, malformed, inconsistent or too small to use."""


class ConfigError(RecsysError):
    """A configuration or hyper-parameter value is invalid."""


class InvalidParameterError(RecsysError):
    """A runtime argument (e.g. k) is invalid."""


class ModelNotFittedError(RecsysError):
    """A model/service was used before being trained."""


class UnknownUserError(RecsysError):
    """The requested user does not exist in the training data."""


class UnknownItemError(RecsysError):
    """The requested item does not exist in the training data."""
