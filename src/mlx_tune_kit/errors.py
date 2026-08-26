"""MLX Tune Kit exceptions."""

__all__ = ["ConfigError", "DatasetError", "MLXTuneKitError", "PlatformError"]


class MLXTuneKitError(Exception):
    """Base error with a user-actionable message."""


class ConfigError(MLXTuneKitError):
    """Raised when a TOML configuration is invalid."""


class DatasetError(MLXTuneKitError):
    """Raised when a dataset cannot be normalized safely."""


class PlatformError(MLXTuneKitError):
    """Raised when MLX training is requested on an unsupported platform."""
