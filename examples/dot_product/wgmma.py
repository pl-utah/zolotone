"""Registry and convenience exports for the reduced WGMMA examples."""

from .wgmma_fp16_e4m3_e5m2 import (
    spec_wgmma_fp16_e4m3_e5m2,
    wgmma_fp16_e4m3_e5m2,
)
from .wgmma_fp32_e4m3_e4m3 import (
    spec_wgmma_fp32_e4m3_e4m3,
    wgmma_fp32_e4m3_e4m3,
)
from .wgmma_fp32_e5m2_e4m3 import (
    spec_wgmma_fp32_e5m2_e4m3,
    wgmma_fp32_e5m2_e4m3,
)


WGMMA_REGISTRY = {
    "wgmma_fp32_e4m3_e4m3": wgmma_fp32_e4m3_e4m3,
    "wgmma_fp32_e5m2_e4m3": wgmma_fp32_e5m2_e4m3,
    "wgmma_fp16_e4m3_e5m2": wgmma_fp16_e4m3_e5m2,
}


__all__ = [
    "WGMMA_REGISTRY",
    "spec_wgmma_fp16_e4m3_e5m2",
    "spec_wgmma_fp32_e4m3_e4m3",
    "spec_wgmma_fp32_e5m2_e4m3",
    "wgmma_fp16_e4m3_e5m2",
    "wgmma_fp32_e4m3_e4m3",
    "wgmma_fp32_e5m2_e4m3",
]
