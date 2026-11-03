"""Owned real intervals backed by the optional native MPFI extension."""

from fractions import Fraction

import _zolotone_mpfi as native


class MPFIInterval:
    """Immutable interval; finite endpoints are exact Python Fractions."""

    __slots__ = ("_handle", "_precision")

    def __init__(self, value=0, precision=128):
        if type(precision) is not int:
            raise TypeError("precision must be an integer number of bits")
        if precision < 2:
            raise ValueError("precision must be at least 2 bits")
        if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
            raise TypeError("expected int, float, or Fraction")
        if not isinstance(value, float):
            ratio = Fraction(value)
            value = f"{ratio.numerator:x}/{ratio.denominator:x}"
        self._handle = native.create(precision, value)
        self._precision = precision

    @classmethod
    def _wrap(cls, handle, precision):
        result = object.__new__(cls)
        result._handle = handle
        result._precision = precision
        return result

    def _endpoint(self, upper):
        value = native.endpoint(self._handle, upper)
        if isinstance(value, float):
            return value
        digits, exponent = value
        return Fraction(int(digits, 2)) * Fraction(2) ** (exponent - len(digits.lstrip("-")))

    def lower(self):
        return self._endpoint(False)

    def upper(self):
        return self._endpoint(True)

    def _unary(self, op):
        return self._wrap(native.unary(op, self._handle), self._precision)

    def _binary(self, op, other):
        if not isinstance(other, MPFIInterval):
            other = MPFIInterval(other, self._precision)
        return self._wrap(native.binary(op, self._handle, other._handle), self._precision)

    def __add__(self, other):
        return self._binary("add", other)

    def __sub__(self, other):
        return self._binary("sub", other)

    def __mul__(self, other):
        return self._binary("mul", other)

    def __truediv__(self, other):
        return self._binary("div", other)

    def __neg__(self):
        return self._unary("neg")

    def __abs__(self):
        return self._unary("abs")

    def __pow__(self, exponent):
        if type(exponent) is not int:
            return self._binary("pow", exponent)
        result = MPFIInterval(1, self._precision)
        base, power = self, abs(exponent)
        while power:
            if power & 1:
                result = result * base
            power >>= 1
            if power:
                base = base._unary("sqr")
        return MPFIInterval(1, self._precision) / result if exponent < 0 else result

    def min(self, other):
        return self._binary("min", other)

    def max(self, other):
        return self._binary("max", other)


class IntervalField:
    """Per-evaluation precision context."""

    def __init__(self, precision):
        if type(precision) is not int:
            raise TypeError("precision must be an integer number of bits")
        if precision < 2:
            raise ValueError("precision must be at least 2 bits")
        self.precision = precision

    def __call__(self, value):
        return MPFIInterval(value, self.precision)

    def pi(self):
        return self(0)._unary("pi")
