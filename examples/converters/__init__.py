"""Conversions between FP32 and every other supported floating-point format."""

from zolotone import (
    BFloat16T,
    E2M1T,
    E4M3FNT,
    E5M2FNUZT,
    E5M2T,
    Float16T,
    Float32T,
)

from .bf16_to_fp32 import bf16_to_fp32
from .fp32_to_bf16 import fp32_to_bf16
from .fp16_to_fp32 import fp16_to_fp32
from .fp32_to_fp16 import fp32_to_fp16
from .e5m2_to_fp32 import e5m2_to_fp32
from .fp32_to_e5m2 import fp32_to_e5m2
from .e5m2fnuz_to_fp32 import e5m2fnuz_to_fp32
from .fp32_to_e5m2fnuz import fp32_to_e5m2fnuz
from .e4m3fn_to_fp32 import e4m3fn_to_fp32
from .fp32_to_e4m3fn import fp32_to_e4m3fn
from .e2m1_to_fp32 import e2m1_to_fp32
from .fp32_to_e2m1 import fp32_to_e2m1

FORMAT_NAMES = (
    "bf16",
    "fp16",
    "fp32",
    "e5m2",
    "e5m2fnuz",
    "e4m3fn",
    "e2m1",
)

CONVERTER_REGISTRY = {
    "bf16_to_fp32": bf16_to_fp32,
    "fp32_to_bf16": fp32_to_bf16,
    "fp16_to_fp32": fp16_to_fp32,
    "fp32_to_fp16": fp32_to_fp16,
    "e5m2_to_fp32": e5m2_to_fp32,
    "fp32_to_e5m2": fp32_to_e5m2,
    "e5m2fnuz_to_fp32": e5m2fnuz_to_fp32,
    "fp32_to_e5m2fnuz": fp32_to_e5m2fnuz,
    "e4m3fn_to_fp32": e4m3fn_to_fp32,
    "fp32_to_e4m3fn": fp32_to_e4m3fn,
    "e2m1_to_fp32": e2m1_to_fp32,
    "fp32_to_e2m1": fp32_to_e2m1,
}
CONVERTER_FORMATS = {
    "bf16_to_fp32": ("bf16", "fp32"),
    "fp32_to_bf16": ("fp32", "bf16"),
    "fp16_to_fp32": ("fp16", "fp32"),
    "fp32_to_fp16": ("fp32", "fp16"),
    "e5m2_to_fp32": ("e5m2", "fp32"),
    "fp32_to_e5m2": ("fp32", "e5m2"),
    "e5m2fnuz_to_fp32": ("e5m2fnuz", "fp32"),
    "fp32_to_e5m2fnuz": ("fp32", "e5m2fnuz"),
    "e4m3fn_to_fp32": ("e4m3fn", "fp32"),
    "fp32_to_e4m3fn": ("fp32", "e4m3fn"),
    "e2m1_to_fp32": ("e2m1", "fp32"),
    "fp32_to_e2m1": ("fp32", "e2m1"),
}
FORMAT_STATIC_TYPES = {
    "bf16": BFloat16T,
    "fp16": Float16T,
    "fp32": Float32T,
    "e5m2": E5M2T,
    "e5m2fnuz": E5M2FNUZT,
    "e4m3fn": E4M3FNT,
    "e2m1": E2M1T,
}

__all__ = [
    "bf16_to_fp32",
    "fp32_to_bf16",
    "fp16_to_fp32",
    "fp32_to_fp16",
    "e5m2_to_fp32",
    "fp32_to_e5m2",
    "e5m2fnuz_to_fp32",
    "fp32_to_e5m2fnuz",
    "e4m3fn_to_fp32",
    "fp32_to_e4m3fn",
    "e2m1_to_fp32",
    "fp32_to_e2m1",
    "CONVERTER_FORMATS",
    "CONVERTER_REGISTRY",
    "FORMAT_NAMES",
    "FORMAT_STATIC_TYPES",
]
