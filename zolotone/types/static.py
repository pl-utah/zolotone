import random


class StaticType:
    def __init__(self):
        self.runtime_val = None
    
    def copy(self) -> "StaticType":
        """Return a fresh StaticType instance with the same shape."""
        new = self._clone_impl()
        new.runtime_val = None if self.runtime_val is None else self.runtime_val.copy()
        return new
    
    def _clone_impl(self) -> "StaticType":
        raise NotImplementedError
    
    def total_bits(self):
        raise NotImplementedError
    
    def to_cpp_type(self, jittable: bool = True) -> str:
        total_bits = self.total_bits()
        if jittable:
            if total_bits <= 8:
                return "uint8_t"
            elif total_bits <= 16:
                return "uint16_t"
            elif total_bits <= 32:
                return "uint32_t"
            elif total_bits <= 64:
                return "uint64_t"
            else:
                raise TypeError("Can not find an ABI-safe type with more than 64 bits in C")  # can use pointer buffers for this
        else:
            return f"ac_uint<{total_bits}>"
    
    def __repr__(self):
        raise NotImplementedError
    
    def __eq__(self, other):
        raise NotImplementedError
    
    def to_spec(self, name, ctx):
        raise NotImplementedError
    
    def random_runtime_value(self, rng: random.Random):
        raise NotImplementedError
    
    def _fingerprint(self):
        from .utils import _fingerprint_value
        fields = tuple(
            sorted(
                (name, _fingerprint_value(value))
                for name, value in vars(self).items()
                if name != "runtime_val"
            )
        )
        runtime_val = None if self.runtime_val is None else self.runtime_val._fingerprint()
        return (type(self).__name__, fields, runtime_val)


class BoolT(StaticType):
    def __init__(self):
        super().__init__()
    
    def total_bits(self):
        return 1
    
    def __repr__(self):
        return f"Bool<1>"
    
    def __str__(self):
        return f"Bool<1>"
    
    def __eq__(self, other):
        return isinstance(other, BoolT)
    
    def _clone_impl(self) -> "QT":
        return BoolT()
    
    def to_spec(self, name, ctx):
        return ctx.fresh_bool(name)
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import Bool
        return Bool(rng.getrandbits(1))


class QT(StaticType):
    def __init__(self, int_bits: int, frac_bits: int):
        super().__init__()
        if int_bits < 0 or frac_bits < 0:
            raise ValueError(
                f"QT bit widths must be non-negative, got int_bits={int_bits}, frac_bits={frac_bits}"
            )
        if int_bits + frac_bits < 1:
            raise ValueError("QT requires at least one total bit")
        self.int_bits, self.frac_bits = int_bits, frac_bits
    
    def total_bits(self):
        return self.int_bits + self.frac_bits
    
    def __repr__(self):
        return f"Q<{self.int_bits},{self.frac_bits}>"
    
    def __str__(self):
        return f"Q<{self.int_bits},{self.frac_bits}>"
    
    def __eq__(self, other):
        return (
            isinstance(other, QT)
            and self.int_bits == other.int_bits
            and self.frac_bits == other.frac_bits
        )
    
    def _clone_impl(self) -> "QT":
        return QT(self.int_bits, self.frac_bits)
    
    def to_spec(self, name, ctx):
        return ctx.fresh_real(name)
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import Q
        return Q(rng.getrandbits(self.total_bits()), self.int_bits, self.frac_bits)


class UQT(StaticType):
    def __init__(self, int_bits: int, frac_bits: int):
        super().__init__()
        if int_bits < 0 or frac_bits < 0:
            raise ValueError(
                f"UQT bit widths must be non-negative, got int_bits={int_bits}, frac_bits={frac_bits}"
            )
        if int_bits + frac_bits < 1:
            raise ValueError("UQT requires at least one total bit")
        self.int_bits, self.frac_bits = int_bits, frac_bits
    
    def total_bits(self):
        return self.int_bits + self.frac_bits
    
    def __repr__(self):
        return f"UQ<{self.int_bits},{self.frac_bits}>"
    
    def __str__(self):
        return f"UQ<{self.int_bits},{self.frac_bits}>"
    
    def __eq__(self, other):
        return (
            isinstance(other, UQT)
            and self.int_bits == other.int_bits
            and self.frac_bits == other.frac_bits
        )
    
    def _clone_impl(self) -> "UQT":
        return UQT(self.int_bits, self.frac_bits)
    
    def to_spec(self, name, ctx):
        variable = ctx.fresh_real(name)
        ctx.assume(variable.eq(abs(variable)))
        return variable
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import UQ
        return UQ(rng.getrandbits(self.total_bits()), self.int_bits, self.frac_bits)


class Float32T(StaticType):
    def __init__(self):
        super().__init__()
        self.sign_bits = 1
        self.mantissa_bits = 23
        self.exponent_bits = 8
    
    def total_bits(self):
        return 32
    
    def __repr__(self):
        return f"Float<32>"
    
    def __str__(self):
        return f"Float<32>"
    
    def __eq__(self, other):
        return (
            isinstance(other, Float32T)
            and self.sign_bits == other.sign_bits
            and self.mantissa_bits == other.mantissa_bits
            and self.exponent_bits == other.exponent_bits
        )
    
    def _clone_impl(self) -> "Float32T":
        return Float32T()
    
    def to_spec(self, name, ctx):
        from ..spec.custom_specs.fp32 import fp32

        return fp32.fresh(name, ctx)
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import Float32
        return Float32(rng.getrandbits(self.total_bits()))


class Float16T(StaticType):
    def __init__(self):
        super().__init__()
        self.sign_bits = 1
        self.mantissa_bits = 10
        self.exponent_bits = 5

    def total_bits(self):
        return 16

    def __repr__(self):
        return "Float<16>"

    def __str__(self):
        return "Float<16>"

    def __eq__(self, other):
        return (
            isinstance(other, Float16T)
            and self.sign_bits == other.sign_bits
            and self.mantissa_bits == other.mantissa_bits
            and self.exponent_bits == other.exponent_bits
        )

    def _clone_impl(self) -> "Float16T":
        return Float16T()

    def to_spec(self, name, ctx):
        from ..spec.custom_specs.fp16 import fp16
        return fp16.fresh(name, ctx)

    def random_runtime_value(self, rng: random.Random):
        from .runtime import Float16
        return Float16(rng.getrandbits(self.total_bits()))


class BFloat16T(StaticType):
    def __init__(self):
        super().__init__()
        self.sign_bits = 1
        self.mantissa_bits = 7
        self.exponent_bits = 8
    
    def total_bits(self):
        return self.sign_bits + self.mantissa_bits + self.exponent_bits
    
    def __repr__(self):
        return f"BFloat<16>"
    
    def __str__(self):
        return f"BFloat<16>"
    
    def __eq__(self, other):
        return (
            isinstance(other, BFloat16T)
            and self.sign_bits == other.sign_bits
            and self.mantissa_bits == other.mantissa_bits
            and self.exponent_bits == other.exponent_bits
        )
    
    def _clone_impl(self) -> "BFloat16T":
        return BFloat16T()
    
    def to_spec(self, name, ctx):
        from ..spec.custom_specs.bf16 import bf16

        return bf16.fresh(name, ctx)
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import BFloat16
        return BFloat16(rng.getrandbits(self.total_bits()))


class E4M3FNT(StaticType):
    """Static type for the finite-only 8-bit E4M3 floating-point format."""

    def __init__(self):
        super().__init__()
        self.sign_bits = 1
        self.exponent_bits = 4
        self.mantissa_bits = 3

    def total_bits(self):
        return 8

    def __repr__(self):
        return "E4M3FN<8>"

    __str__ = __repr__

    def __eq__(self, other):
        return isinstance(other, E4M3FNT)

    def _clone_impl(self) -> "E4M3FNT":
        return E4M3FNT()

    def to_spec(self, name, ctx):
        from ..spec.custom_specs.e4m3fn import e4m3fn

        return e4m3fn.fresh(name, ctx)

    def random_runtime_value(self, rng: random.Random):
        from .runtime import E4M3FN

        return E4M3FN(rng.getrandbits(self.total_bits()))


class TupleT(StaticType):
    def __init__(self, *args: StaticType):
        super().__init__()
        for x in args:
            if not isinstance(x, StaticType):
                raise TypeError(f"TupleT can not contain non-StaticType, given: {x}")
        if len(args) == 0:
            raise ValueError("tuple can not be empty")
        self.args = args
    
    def total_bits(self):
        return sum([x.total_bits() for x in self.args])
    
    def __repr__(self):
        return f"Tuple<{', '.join([repr(x) for x in self.args])}>"
    
    def __str__(self):
        return f"Tuple<{', '.join([repr(x) for x in self.args])}>"
    
    def __eq__(self, other):
        return (
            isinstance(other, TupleT)
            and len(self.args) == len(other.args)
            and all([arg1 == arg2 for arg1, arg2 in zip(self.args, other.args)])
        )
    
    def _clone_impl(self) -> "TupleT":
        return TupleT(*[arg.copy() for arg in self.args])
    
    def to_spec(self, name, ctx):
        return tuple(x.to_spec(name=f"{name}_{i}", ctx=ctx) for i, x in enumerate(self.args))
    
    def to_cpp_type(self, jittable: bool = True) -> str:
        return f"std::array<uint64_t, {len(self.args)}>" if jittable else f"std::tuple<{', '.join(arg.to_cpp_type(jittable=jittable) for arg in self.args)}>"
    
    def random_runtime_value(self, rng: random.Random):
        from .runtime import Tuple
        return Tuple(*[arg.random_runtime_value(rng) for arg in self.args])
