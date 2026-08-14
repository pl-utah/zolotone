# FP32 converters

This package contains conversions between FP32 and every other floating-point
format supported by Zolotone. Converters are imported directly from the
package:

```python
from examples.converters import e5m2_to_fp32, fp32_to_e5m2
```

| format | to FP32 | from FP32 |
| --- | --- | --- |
| `bf16` | `bf16_to_fp32` | `fp32_to_bf16` |
| `fp16` | `fp16_to_fp32` | `fp32_to_fp16` |
| `e5m2` | `e5m2_to_fp32` | `fp32_to_e5m2` |
| `e5m2fnuz` | `e5m2fnuz_to_fp32` | `fp32_to_e5m2fnuz` |
| `e4m3fn` | `e4m3fn_to_fp32` | `fp32_to_e4m3fn` |
| `ue4m3` | `ue4m3_to_fp32` | `fp32_to_ue4m3` |
| `e2m1` | `e2m1_to_fp32` | `fp32_to_e2m1` |

Each module contains its complete specification and implementation. Widening
to FP32 is exact: normal values are rebased and packed directly, while source
subnormals are normalized before packing. Conversions from FP32 use the
destination encoder for round-to-nearest-even, underflow, overflow, and finite
saturation.

Special values follow these rules:

- signed zero is preserved except when converting to unsigned-zero
  `e5m2fnuz` or `ue4m3`;
- NaN is canonicalized when the destination supports NaN;
- infinity stays signed infinity when the destination supports it;
- infinity saturates to signed maximum finite in signed finite-only formats;
- both signs of infinity saturate to positive 448 in `ue4m3`, and finite
  negative FP32 values are converted by magnitude;
- FP32 NaN converts to positive maximum finite in E2M1, matching documented
  [CUDA E2M1 conversion behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html).

Running any module checks determinism and its specification, then emits JIT and
non-JIT C++ headers under `examples/c_models/`:

```sh
python -m examples.converters.e5m2_to_fp32
```
