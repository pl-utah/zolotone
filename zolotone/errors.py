"""Public exception types raised by Zolotone."""


class ZolotoneError(Exception):
    """Base class for Zolotone-specific failures."""


class InfeasibleError(ZolotoneError, TypeError):
    """A specification cannot be satisfied under its declared contract."""


class MissingError(ZolotoneError, TypeError):
    """Required type information or lowering support is unavailable."""


__all__ = ["ZolotoneError", "InfeasibleError", "MissingError"]
