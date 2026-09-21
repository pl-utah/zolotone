"""Public exception types raised by Zolotone."""


class ZolotoneError(Exception):
    """Base class for Zolotone-specific failures."""


class InfeasibleError(ZolotoneError, TypeError):
    """A valid specification cannot fit its requested output format."""


class MissingError(ZolotoneError, TypeError):
    """Required type information or lowering support is unavailable."""


__all__ = ["ZolotoneError", "InfeasibleError", "MissingError"]
