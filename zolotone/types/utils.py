from .descriptors import DataType
from .values import RuntimeValue

def _fingerprint_value(value):
    if isinstance(value, RuntimeValue):
        return value._fingerprint()
    if isinstance(value, DataType):
        return value._fingerprint()
    if isinstance(value, tuple):
        return tuple(_fingerprint_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_fingerprint_value(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_fingerprint_value(key), _fingerprint_value(val))
                for key, val in value.items()
            )
        )
    return value
