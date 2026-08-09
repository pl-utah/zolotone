import random
import time

from .static import BFloat16T, BoolT, E4M3FNT, Float16T, Float32T, QT, TupleT, UQT


class RuntimeType:
    def to_spec(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def random_generator(cls):
        raise NotImplementedError
    
    def static_type(self):
        raise NotImplementedError
    
    def copy(self):
        raise NotImplementedError
    
    def total_bits(self):
        raise NotImplementedError
    
    def __eq__(self, other):
        raise NotImplementedError

    def _fingerprint(self):
        from .utils import _fingerprint_value
        return (
            type(self).__name__,
            tuple(
                sorted(
                    (name, _fingerprint_value(value))
                    for name, value in vars(self).items()
                )
            ),
        )


# TODO: Tuple should have a default self.val field
class Tuple(RuntimeType):
    def __init__(self, *args: RuntimeType):
        if not args:
            raise TypeError("Tuple cannot be empty")
        for x in args:
            if not isinstance(x, RuntimeType):
                raise TypeError("Tuple must contain RuntimeType instances")
        
        self.args = args
    
    def to_val(self):
        return tuple(x.to_val() for x in self.args)
    
    def to_spec(self, ctx):
        return tuple(x.to_spec(ctx) for x in self.args)
    
    def __str__(self):
        return f"Tuple[{', '.join([str(x) for x in self.args])}]"
    
    def static_type(self):
        return TupleT(*[x.static_type() for x in self.args])
    
    def total_bits(self):
        return sum([x.total_bits() for x in self.args])
    
    def copy(self, val=None):
        if val is None:
            return Tuple(*[x.copy() for x in self.args])
        
        if isinstance(val, tuple) and all(isinstance(x, RuntimeType) for x in val):
            assert len(val) == len(self.args), "Tuple lengths do not match"
            assert all(
                x.static_type() == y.static_type()
                for x, y in zip(val, self.args)
            ), "Tuple's internal types do not match"
            return Tuple(*[x.copy() for x in val])

        expected_types = tuple(x.static_type() for x in self.args)
        raise ValueError(
            f"Wrong val passed, expected tuple with length {len(self.args)} "
            f"and types {expected_types}"
        )
    
    def __eq__(self, other):
        return (
            isinstance(other, Tuple)
            and all([x == y for x, y in zip(self.args, other.args)])
        )


class Bool(RuntimeType):
    def __init__(self, val: int):
        if val not in (0, 1):
            raise ValueError(f"Bool value must be 0 or 1, got {val}")
        self.val = val
        
    def __str__(self):
        return f"Bool({self.val})"
    
    def to_val(self):
        return self.val == 1
    
    def to_spec(self, ctx):
        return ctx.bool_val(self.to_val())
    
    def static_type(self):
        return BoolT()
    
    def copy(self, val=None):
        if val is None:
            val = self.val
        return Bool(val)
    
    def total_bits(self):
        return 1
    
    def __eq__(self, other):
        return (
            isinstance(other, Bool)
            and other.val == self.val
        )
    

class Q(RuntimeType):
    """Signed fixed-point type."""
    def __init__(self, val: int, int_bits: int, frac_bits: int):
        self.val, self.int_bits, self.frac_bits = val, int_bits, frac_bits
        
        if self.int_bits < 0:
            raise ValueError(f"Q integer bits must be non-negative, got {self.int_bits}")
        if self.frac_bits < 0:
            raise ValueError(f"Q fractional bits must be non-negative, got {self.frac_bits}")
        if self.int_bits + self.frac_bits < 1:
            raise ValueError(
                f"Q requires at least one total bit, got {self.int_bits + self.frac_bits}"
            )
        if not (0 <= self.val < (1 << self.total_bits())):
            raise ValueError(
                f"Q value {self.val} does not fit into {self.total_bits()} bits"
            )
    
    def __str__(self):
        return f"Q{self.int_bits}.{self.frac_bits}({str(self.to_val())})"
    
    def to_val(self):
        sign_bit = self.val >> (self.total_bits() - 1)
        if sign_bit == 1:
            signed_val = self.val - (sign_bit << self.total_bits())
            return float(signed_val) / (2 ** self.frac_bits)
        else:
            return float(self.val) / (2 ** self.frac_bits)
    
    def to_spec(self, ctx):
        return ctx.real_val(self.to_val())
    
    def static_type(self):
        return QT(self.int_bits, self.frac_bits)
    
    def copy(self, val=None):
        if val is None:
            val = self.val
        return Q(val, self.int_bits, self.frac_bits)
    
    def total_bits(self):
        return self.int_bits + self.frac_bits
    
    # Custom methods
    @staticmethod
    def from_int(x: int):
        if x < 0:
            magnitude = abs(x)
            int_bits = max(2, (magnitude - 1).bit_length() + 1)
            val = (1 << int_bits) + x
        else:
            int_bits = max(2, x.bit_length() + 1)
            val = x
        return Q(val, int_bits, 0)
    
    @staticmethod
    def from_float(x: float, target_int: int, target_frac: int):
        W = target_int + target_frac
        scale = 1 << target_frac

        # Float -> bits
        q = int(round(x * scale))

        # Clamps overflow/underflow
        min_q = -(1 << (W - 1))
        max_q =  (1 << (W - 1)) - 1
        if q < min_q:
            q = min_q
        elif q > max_q:
            q = max_q

        # encode to raw two's-complement bits
        bits = q & ((1 << W) - 1)

        return Q(bits, target_int, target_frac)
    
    def __eq__(self, other):
        return (
            isinstance(other, Q)
            and other.int_bits == self.int_bits 
            and other.frac_bits == self.frac_bits 
            and other.val == self.val
        )


class UQ(RuntimeType):
    """Unsigned fixed-point type."""
    def __init__(self, val: int, int_bits: int, frac_bits: int):
        total_bits = int_bits + frac_bits
        
        if int_bits < 0:
            raise ValueError(f"UQ integer bits must be non-negative, got {int_bits}")
        if frac_bits < 0:
            raise ValueError(f"UQ fractional bits must be non-negative, got {frac_bits}")
        if int_bits == 0 and frac_bits == 0:
            raise ValueError("UQ requires at least one total bit")
        if not (0 <= val < (1 << total_bits)):
            raise ValueError(
                f"Value {val} requires {max(1, val.bit_length())} bits, "
                f"but only {total_bits} provided ({int_bits}+{frac_bits})"
            )
        
        self.val, self.int_bits, self.frac_bits = val, int_bits, frac_bits
    
    def __str__(self):
        return f"UQ{self.int_bits}.{self.frac_bits}({str(self.to_val())})"
    
    def to_val(self):
        return float(self.val) / (2 ** self.frac_bits)
    
    def to_spec(self, ctx):
        return ctx.real_val(self.to_val())
    
    def static_type(self):
        return UQT(self.int_bits, self.frac_bits)
    
    def copy(self, val=None):
        if val is None:
            val = self.val
        return UQ(val, self.int_bits, self.frac_bits)
    
    def total_bits(self):
        return self.int_bits + self.frac_bits
    
    # Custom methods
    @staticmethod
    def from_int(x: int):
        return UQ(x, max(1, x.bit_length()), 0)

    @staticmethod
    def from_float(x: float, target_int: int, target_frac: int):
        W = target_int + target_frac
        scale = 1 << target_frac

        # Float -> bits
        q = int(round(x * scale))

        # Unsigned fixed-point clamps to the representable non-negative range.
        min_q = 0
        max_q = (1 << W) - 1
        if q < min_q:
            q = min_q
        elif q > max_q:
            q = max_q

        return UQ(q, target_int, target_frac)
    
    def __eq__(self, other):
        return (
            isinstance(other, UQ)
            and other.int_bits == self.int_bits 
            and other.frac_bits == self.frac_bits 
            and other.val == self.val
        )


class Float32(RuntimeType):
    """Single-precision floating-point format, IEEE754-1985"""
    mantissa_bits = 23
    exponent_bits = 8
    exponent_bias = 127
    inf_code = 255
    sub_code = 0
    nan_code = 255
    zero_code = 0
    
    def __init__(self, val: int):
        if not isinstance(val, int):
            raise TypeError(f"Float32 expects packed bits as int, got {type(val).__name__}")
        if not (0 <= val < (1 << self.total_bits())):
            raise ValueError(f"Float32 packed bits out of range: {val}")
        self.val = val

    @classmethod
    def from_fields(cls, sign: int, exponent: int, mantissa: int):
        if sign not in (0, 1):
            raise ValueError(f"Float32 sign must be 0 or 1, got {sign}")
        if not (0 <= mantissa < (1 << cls.mantissa_bits)):
            raise ValueError(f"Float32 mantissa out of range: {mantissa}")
        if not (0 <= exponent < (1 << cls.exponent_bits)):
            raise ValueError(f"Float32 exponent out of range: {exponent}")
        return cls(
            (sign << (cls.exponent_bits + cls.mantissa_bits))
            | (exponent << cls.mantissa_bits)
            | mantissa
        )

    @property
    def sign(self):
        return (self.val >> (self.exponent_bits + self.mantissa_bits)) & 1

    @property
    def exponent(self):
        return (self.val >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)

    @property
    def mantissa(self):
        return self.val & ((1 << self.mantissa_bits) - 1)

    @property
    def significand(self):
        return self.mantissa
    
    def __str__(self):
        return f"Float({str(self.to_val())})"
    
    def to_val(self):
        """Converts to actual floating-point value (IEEE754-style)."""
        # Infinity
        if self.exponent == self.inf_code and self.mantissa == 0:
            return float('-inf') if self.sign == 1 else float('inf')
        # NaN
        elif self.exponent == self.nan_code and self.mantissa != 0:
            return float('nan')
        # Zero/subnormal
        elif self.exponent == 0:
            # Subnormal numbers (no implicit 1)
            frac = self.mantissa / (2 ** self.mantissa_bits)
            exp_val = 1 - self.exponent_bias
            return float((-1) ** self.sign * frac * (2 ** exp_val))
        # Normal
        else:
            frac = 1.0 + self.mantissa / (2 ** self.mantissa_bits)
            exp_val = self.exponent - self.exponent_bias
            return float((-1) ** self.sign * frac * (2 ** exp_val))

    def to_spec(self, ctx):
        from ..spec.custom_specs.fp32 import fp32

        if self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 0:
            return fp32.inf(ctx)
        elif self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 1:
            return fp32.ninf(ctx)
        elif self.exponent == self.nan_code and self.mantissa != 0:
            return fp32.nan(ctx)
        elif self.exponent == 0 and self.mantissa == 0 and self.sign == 1:
            return fp32.nzero(ctx)
        elif self.exponent == 0 and self.mantissa == 0 and self.sign == 0:
            return fp32.zero(ctx)
        elif self.exponent == 0 and self.mantissa != 0:
             return fp32(
                 value=ctx.real_val(self.to_val()),
                 sign=ctx.real_val(self.sign),
                 exponent=ctx.real_val(self.exponent),
                 mantissa=ctx.real_val(self.mantissa),
                 is_norm=ctx.bool_val(False),
                 is_sub=ctx.bool_val(True),
                 is_zero=ctx.bool_val(False),
                 is_inf=ctx.bool_val(False),
                 is_nan=ctx.bool_val(False),
             )
        else:
             return fp32(
                 value=ctx.real_val(self.to_val()),
                 sign=ctx.real_val(self.sign),
                 exponent=ctx.real_val(self.exponent),
                 mantissa=ctx.real_val(self.mantissa),
                 is_norm=ctx.bool_val(True),
                 is_sub=ctx.bool_val(False),
                 is_zero=ctx.bool_val(False),
                 is_inf=ctx.bool_val(False),
                 is_nan=ctx.bool_val(False),
             )
       
    
    def static_type(self):
        return Float32T()
    
    @classmethod
    def nInf(cls):
        return cls.from_fields(1, cls.inf_code, 0)
    
    @classmethod
    def Inf(cls):
        return cls.from_fields(0, cls.inf_code, 0)
    
    @classmethod
    def nZero(cls):
        return cls.from_fields(1, cls.zero_code, 0)
    
    @classmethod
    def Zero(cls):
        return cls.from_fields(0, cls.zero_code, 0)
    
    @classmethod
    def NaN(cls, payload: int | None = None):
        if payload is None:
            payload = 1 << (cls.mantissa_bits - 1)
        if not isinstance(payload, int):
            raise TypeError(f"Float32 NaN payload must be int, got {type(payload).__name__}")
        if not (1 <= payload < (1 << cls.mantissa_bits)):
            raise ValueError(
                f"Float32 NaN payload must fit in {cls.mantissa_bits} mantissa bits and be non-zero, got {payload}"
            )
        return cls.from_fields(0, cls.nan_code, payload)
    
    def copy(self, val=None):
        if val is None:
            val = self.val
        return Float32(val)
    
    def total_bits(self):
        return 32
    
    def __eq__(self, other):
        return (
            isinstance(other, Float32)
            and self.val == other.val
        )


class Float16(RuntimeType):
    """IEEE-754 binary16 format: 1 sign, 5 exponent, 10 mantissa bits."""

    mantissa_bits = 10
    exponent_bits = 5
    exponent_bias = 15
    inf_code = 31
    sub_code = 0
    nan_code = 31
    zero_code = 0

    def __init__(self, val: int):
        if not isinstance(val, int):
            raise TypeError(
                f"Float16 expects packed bits as int, got {type(val).__name__}"
            )
        if not (0 <= val < (1 << self.total_bits())):
            raise ValueError(
                f"Float16 packed bits must fit in {self.total_bits()} bits, got {val}"
            )
        self.val = val

    @classmethod
    def from_fields(cls, sign: int, exponent: int, mantissa: int):
        if sign not in (0, 1):
            raise ValueError(f"Float16 sign must be 0 or 1, got {sign}")
        if not (0 <= mantissa < (1 << cls.mantissa_bits)):
            raise ValueError(f"Float16 mantissa out of range: {mantissa}")
        if not (0 <= exponent < (1 << cls.exponent_bits)):
            raise ValueError(f"Float16 exponent out of range: {exponent}")
        return cls(
            (sign << (cls.exponent_bits + cls.mantissa_bits))
            | (exponent << cls.mantissa_bits)
            | mantissa
        )

    @property
    def sign(self):
        return (self.val >> (self.exponent_bits + self.mantissa_bits)) & 1

    @property
    def exponent(self):
        return (self.val >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)

    @property
    def mantissa(self):
        return self.val & ((1 << self.mantissa_bits) - 1)

    @property
    def significand(self):
        return self.mantissa

    def __str__(self):
        return f"Float16({str(self.to_val())})"

    def to_val(self):
        """Convert the packed binary16 value to a Python float."""

        if self.exponent == self.inf_code and self.mantissa == 0:
            return float("-inf") if self.sign else float("inf")
        if self.exponent == self.nan_code and self.mantissa != 0:
            return float("nan")
        if self.exponent == 0:
            fraction = self.mantissa / (2 ** self.mantissa_bits)
            exponent = 1 - self.exponent_bias
            return float((-1) ** self.sign * fraction * (2 ** exponent))

        fraction = 1.0 + self.mantissa / (2 ** self.mantissa_bits)
        exponent = self.exponent - self.exponent_bias
        return float((-1) ** self.sign * fraction * (2 ** exponent))

    def to_spec(self, ctx):
        from ..spec.custom_specs.fp16 import fp16

        if self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 0:
            return fp16.inf(ctx)
        if self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 1:
            return fp16.ninf(ctx)
        if self.exponent == self.nan_code and self.mantissa != 0:
            return fp16.nan(ctx)
        if self.exponent == 0 and self.mantissa == 0 and self.sign == 1:
            return fp16.nzero(ctx)
        if self.exponent == 0 and self.mantissa == 0 and self.sign == 0:
            return fp16.zero(ctx)

        return fp16(
            value=ctx.real_val(self.to_val()),
            sign=ctx.real_val(self.sign),
            exponent=ctx.real_val(self.exponent),
            mantissa=ctx.real_val(self.mantissa),
            is_norm=ctx.bool_val(self.exponent != 0),
            is_sub=ctx.bool_val(self.exponent == 0),
            is_zero=ctx.bool_val(False),
            is_inf=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )

    def static_type(self):
        return Float16T()

    @classmethod
    def nInf(cls):
        return cls.from_fields(sign=1, exponent=cls.inf_code, mantissa=0)

    @classmethod
    def Inf(cls):
        return cls.from_fields(sign=0, exponent=cls.inf_code, mantissa=0)

    @classmethod
    def nZero(cls):
        return cls.from_fields(sign=1, exponent=cls.zero_code, mantissa=0)

    @classmethod
    def Zero(cls):
        return cls.from_fields(sign=0, exponent=cls.zero_code, mantissa=0)

    @classmethod
    def NaN(cls, payload: int | None = None):
        if payload is None:
            payload = 1 << (cls.mantissa_bits - 1)
        if not isinstance(payload, int):
            raise TypeError(
                f"Float16 NaN payload must be int, got {type(payload).__name__}"
            )
        if not (1 <= payload < (1 << cls.mantissa_bits)):
            raise ValueError(
                f"Float16 NaN payload must fit in {cls.mantissa_bits} mantissa "
                f"bits and be non-zero, got {payload}"
            )
        return cls.from_fields(sign=0, exponent=cls.nan_code, mantissa=payload)

    def copy(self, val=None):
        if val is None:
            val = self.val
        return Float16(val)

    def total_bits(self):
        return 16

    def __eq__(self, other):
        return isinstance(other, Float16) and self.val == other.val


class BFloat16(RuntimeType):
    """Brain Floating Point 16-bit (bfloat16) format — 1 sign, 8 exponent, 7 mantissa bits."""
    mantissa_bits = 7
    exponent_bits = 8
    exponent_bias = 127
    inf_code = 255
    sub_code = 0
    nan_code = 255
    zero_code = 0
    
    def __init__(self, val: int):
        if not isinstance(val, int):
            raise TypeError(f"BFloat16 expects packed bits as int, got {type(val).__name__}")
        if not (0 <= val < (1 << self.total_bits())):
            raise ValueError(f"BFloat16 packed bits must fit in {self.total_bits()} bits, got {val}")
        self.val = val

    @classmethod
    def from_fields(cls, sign: int, mantissa: int, exponent: int):
        if sign not in (0, 1):
            raise ValueError(f"BFloat16 sign must be 0 or 1, got {sign}")
        if not (0 <= mantissa < (1 << cls.mantissa_bits)):
            raise ValueError(f"BFloat16 mantissa out of range: {mantissa}")
        if not (0 <= exponent < (1 << cls.exponent_bits)):
            raise ValueError(f"BFloat16 exponent out of range: {exponent}")
        return cls(
            (sign << (cls.exponent_bits + cls.mantissa_bits))
            | (exponent << cls.mantissa_bits)
            | mantissa
        )

    @property
    def sign(self):
        return (self.val >> (self.exponent_bits + self.mantissa_bits)) & 1

    @property
    def exponent(self):
        return (self.val >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)

    @property
    def mantissa(self):
        return self.val & ((1 << self.mantissa_bits) - 1)

    @property
    def significand(self):
        return self.mantissa
    
    def __str__(self):
        return f"BFloat16({str(self.to_val())})"
    
    def to_val(self):
        """Converts to IEEE754-style float value."""
        if self.exponent == self.inf_code and self.mantissa == 0:
            return float("-inf") if self.sign else float("inf")
        if self.exponent == self.nan_code and self.mantissa != 0:
            return float("nan")
        if self.exponent == 0:
            fraction = self.mantissa / (2 ** self.mantissa_bits)
            exponent = 1 - self.exponent_bias
            return float((-1) ** self.sign * fraction * (2 ** exponent))

        fraction = 1.0 + self.mantissa / (2 ** self.mantissa_bits)
        exponent = self.exponent - self.exponent_bias
        return float((-1) ** self.sign * fraction * (2 ** exponent))
    
    def to_spec(self, ctx):
        from ..spec.custom_specs.bf16 import bf16

        if self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 0:
            return bf16.inf(ctx)
        if self.exponent == self.inf_code and self.mantissa == 0 and self.sign == 1:
            return bf16.ninf(ctx)
        if self.exponent == self.nan_code and self.mantissa != 0:
            return bf16.nan(ctx)
        if self.exponent == 0 and self.mantissa == 0 and self.sign == 1:
            return bf16.nzero(ctx)
        if self.exponent == 0 and self.mantissa == 0 and self.sign == 0:
            return bf16.zero(ctx)

        return bf16(
            value=ctx.real_val(self.to_val()),
            sign=ctx.real_val(self.sign),
            exponent=ctx.real_val(self.exponent),
            mantissa=ctx.real_val(self.mantissa),
            is_norm=ctx.bool_val(self.exponent != 0),
            is_sub=ctx.bool_val(self.exponent == 0),
            is_zero=ctx.bool_val(False),
            is_inf=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )
    
    def static_type(self):
        return BFloat16T()

    @classmethod
    def nInf(cls):
        return cls.from_fields(sign=1, exponent=cls.inf_code, mantissa=0)

    @classmethod
    def Inf(cls):
        return cls.from_fields(sign=0, exponent=cls.inf_code, mantissa=0)

    @classmethod
    def nZero(cls):
        return cls.from_fields(sign=1, exponent=cls.zero_code, mantissa=0)

    @classmethod
    def Zero(cls):
        return cls.from_fields(sign=0, exponent=cls.zero_code, mantissa=0)

    @classmethod
    def NaN(cls, payload: int | None = None):
        if payload is None:
            payload = 1 << (cls.mantissa_bits - 1)
        if not isinstance(payload, int):
            raise TypeError(f"BFloat16 NaN payload must be int, got {type(payload).__name__}")
        if not (1 <= payload < (1 << cls.mantissa_bits)):
            raise ValueError(
                f"BFloat16 NaN payload must fit in {cls.mantissa_bits} mantissa "
                f"bits and be non-zero, got {payload}"
            )
        return cls.from_fields(sign=0, exponent=cls.nan_code, mantissa=payload)
    
    @classmethod
    def random_generator(cls, seed = None, shared_exponent_bits: int = 0):
        if seed is None:
            seed = int(time.time())
        if not (0 <= shared_exponent_bits <= cls.exponent_bits):
            raise ValueError(
                f"shared_exponent_bits must be between 0 and {cls.exponent_bits}, "
                f"got {shared_exponent_bits}"
            )
        
        rnd = random.Random(seed)
        unshared_exponent_bits = cls.exponent_bits - shared_exponent_bits
        shared_exp = rnd.getrandbits(shared_exponent_bits) << unshared_exponent_bits
        
        def gen():
            sign = rnd.getrandbits(1)
            mantissa = rnd.getrandbits(cls.mantissa_bits)
            exponent = shared_exp + rnd.getrandbits(unshared_exponent_bits)
            return cls.from_fields(sign=sign, mantissa=mantissa, exponent=exponent)
        
        def gen_shared_exp():
            nonlocal shared_exp
            shared_exp = rnd.getrandbits(shared_exponent_bits) << unshared_exponent_bits
            return shared_exp
        
        return gen, gen_shared_exp
    
    def copy(self, val=None):
        if val is None:
            val = self.val
        return BFloat16(val)
    
    def total_bits(self):
        return 16
    
    def __eq__(self, other):
        return (
            isinstance(other, BFloat16)
            and self.val == other.val
        )


class E4M3FN(RuntimeType):
    """Finite-only E4M3 format with signed zero, subnormals, and NaN."""

    sign_bits = 1
    exponent_bits = 4
    mantissa_bits = 3
    exponent_bias = 7
    sub_code = 0
    zero_code = 0
    nan_code = 15
    nan_mantissa = 7
    max_finite_code = 15
    max_finite_mantissa = 6

    def __init__(self, val: int):
        if not isinstance(val, int):
            raise TypeError(
                f"E4M3FN expects packed bits as int, got {type(val).__name__}"
            )
        if not (0 <= val < (1 << self.total_bits())):
            raise ValueError(
                f"E4M3FN packed bits must fit in {self.total_bits()} bits, got {val}"
            )
        self.val = val

    @classmethod
    def from_fields(cls, sign: int, exponent: int, mantissa: int):
        if sign not in (0, 1):
            raise ValueError(f"E4M3FN sign must be 0 or 1, got {sign}")
        if not isinstance(exponent, int) or not (
            0 <= exponent < (1 << cls.exponent_bits)
        ):
            raise ValueError(f"E4M3FN exponent out of range: {exponent}")
        if not isinstance(mantissa, int) or not (
            0 <= mantissa < (1 << cls.mantissa_bits)
        ):
            raise ValueError(f"E4M3FN mantissa out of range: {mantissa}")
        return cls(
            (sign << (cls.exponent_bits + cls.mantissa_bits))
            | (exponent << cls.mantissa_bits)
            | mantissa
        )

    @property
    def sign(self):
        return (self.val >> (self.exponent_bits + self.mantissa_bits)) & 1

    @property
    def exponent(self):
        return (self.val >> self.mantissa_bits) & ((1 << self.exponent_bits) - 1)

    @property
    def mantissa(self):
        return self.val & ((1 << self.mantissa_bits) - 1)

    @property
    def significand(self):
        return self.mantissa

    @property
    def is_nan(self):
        return self.exponent == self.nan_code and self.mantissa == self.nan_mantissa

    def __str__(self):
        return f"E4M3FN({self.to_val()})"

    def to_val(self):
        if self.is_nan:
            return float("nan")
        sign = -1.0 if self.sign else 1.0
        if self.exponent == 0:
            return float(
                sign
                * (self.mantissa / (2 ** self.mantissa_bits))
                * (2 ** (1 - self.exponent_bias))
            )
        return float(
            sign
            * (1.0 + self.mantissa / (2 ** self.mantissa_bits))
            * (2 ** (self.exponent - self.exponent_bias))
        )

    def to_spec(self, ctx):
        from ..spec.custom_specs.e4m3fn import e4m3fn

        if self.is_nan:
            return e4m3fn.nan(ctx)
        if self.exponent == 0 and self.mantissa == 0:
            return e4m3fn.nzero(ctx) if self.sign else e4m3fn.zero(ctx)
        return e4m3fn(
            value=ctx.real_val(self.to_val()),
            sign=ctx.real_val(self.sign),
            exponent=ctx.real_val(self.exponent),
            mantissa=ctx.real_val(self.mantissa),
            is_norm=ctx.bool_val(self.exponent != 0),
            is_sub=ctx.bool_val(self.exponent == 0),
            is_zero=ctx.bool_val(False),
            is_nan=ctx.bool_val(False),
        )

    def static_type(self):
        return E4M3FNT()

    @classmethod
    def Zero(cls):
        return cls.from_fields(0, cls.zero_code, 0)

    @classmethod
    def nZero(cls):
        return cls.from_fields(1, cls.zero_code, 0)

    @classmethod
    def NaN(cls):
        return cls.from_fields(0, cls.nan_code, cls.nan_mantissa)

    @classmethod
    def random_generator(cls, seed=None, shared_exponent_bits: int = 0):
        if seed is None:
            seed = int(time.time())
        if not (0 <= shared_exponent_bits <= cls.exponent_bits):
            raise ValueError(
                f"shared_exponent_bits must be between 0 and {cls.exponent_bits}, "
                f"got {shared_exponent_bits}"
            )
        rnd = random.Random(seed)
        unshared_exponent_bits = cls.exponent_bits - shared_exponent_bits
        shared_exponent = (
            rnd.getrandbits(shared_exponent_bits) << unshared_exponent_bits
        )

        def gen():
            return cls.from_fields(
                sign=rnd.getrandbits(1),
                exponent=(
                    shared_exponent + rnd.getrandbits(unshared_exponent_bits)
                ),
                mantissa=rnd.getrandbits(cls.mantissa_bits),
            )

        def gen_shared_exp():
            nonlocal shared_exponent
            shared_exponent = (
                rnd.getrandbits(shared_exponent_bits) << unshared_exponent_bits
            )
            return shared_exponent

        return gen, gen_shared_exp

    def copy(self, val=None):
        return E4M3FN(self.val if val is None else val)

    def total_bits(self):
        return 8

    def __eq__(self, other):
        return isinstance(other, E4M3FN) and self.val == other.val
