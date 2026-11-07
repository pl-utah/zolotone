"""Data-only serialization for scalar spec ASTs and their associated values.

The encoding is shared by SpecNode and verification certificates. Decoding uses
an explicit constructor allowlist, never imports serialized names, and preserves
exact rational values. Floating-point AST representations are not supported yet.
"""

from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction


def encode_spec_value(value):
    """Canonical, data-only encoding; never execute serialized Python objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Fraction)):
        return {"rational": str(Fraction(value))}
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if isinstance(value, type):
        return {"type": f"{value.__module__}.{value.__qualname__}"}
    if isinstance(value, (tuple, list)):
        return [encode_spec_value(item) for item in value]
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            "fields": {field.name: encode_spec_value(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"Cannot serialize specification value: {type(value).__name__}")


def decode_spec_value(value):
    """Decode only explicitly supported data; never import names from JSON."""
    from . import spec_ast as ast
    from ..types import Bool, Q, UQ

    # Keep this list explicit: adding a Python class must not automatically make
    # it constructible from a specification.
    nodes = (
        ast.RealVar, ast.BoolVar, ast.RealLit, ast.BoolLit, ast.Round,
        ast.Add, ast.Sub, ast.Mul, ast.Neg, ast.Abs, ast.Pi,
        ast.Sin, ast.Cos, ast.Exp, ast.Exp2, ast.Log, ast.Log2, ast.Sqrt,
        ast.Tanh, ast.Pow2, ast.Cbrt, ast.Expm1, ast.Log10, ast.Log1p, ast.Logb,
        ast.Tan, ast.Asin, ast.Acos, ast.Atan, ast.Sinh, ast.Cosh,
        ast.Asinh, ast.Acosh, ast.Atanh, ast.Erf, ast.Erfc,
        ast.Pow, ast.Max, ast.Min, ast.If, ast.Eq, ast.NotEq,
        ast.Lt, ast.Le, ast.Gt, ast.Ge, ast.BoolEq, ast.BoolNe,
        ast.Not, ast.Or, ast.And, ast._CaseEntry, ast._CasePartition, Q, UQ, Bool,
    )
    constructors = {node.__name__: node for node in nodes}
    types = {f"{dtype.__module__}.{dtype.__qualname__}": dtype for dtype in (Q, UQ, Bool)}

    def decode(data):
        if data is None or type(data) in (str, bool, int):
            return data
        if type(data) is list:
            return [decode(item) for item in data]
        if type(data) is not dict:
            raise ValueError("Invalid specification encoded value")
        if set(data) == {"rational"} and type(data["rational"]) is str:
            return Fraction(data["rational"])
        if set(data) == {"enum", "value"} and data["enum"] == "RoundingMode":
            return ast.RoundingMode(data["value"])
        if set(data) == {"type"} and type(data["type"]) is str:
            if data["type"] in types:
                return types[data["type"]]
        if set(data) == {"node", "fields"} and type(data["node"]) is str:
            constructor = constructors.get(data["node"])
            encoded_fields = data["fields"]
            if constructor is None or type(encoded_fields) is not dict:
                raise ValueError("Unknown specification node or invalid fields")
            if set(encoded_fields) != {field.name for field in fields(constructor)}:
                raise ValueError(f"Invalid specification fields for {data['node']}")
            kwargs = {name: decode(item) for name, item in encoded_fields.items()}
            if constructor in (ast.RealVar, ast.BoolVar):
                if type(kwargs["name"]) is not str or not kwargs["name"]:
                    raise ValueError("Specification variable names must be nonempty strings")
            elif constructor is ast.RealLit:
                if type(kwargs["value"]) not in (int, Fraction):
                    raise ValueError("Specification real literals must be exact numbers")
            elif constructor is ast.BoolLit:
                if type(kwargs["value"]) is not bool:
                    raise ValueError("Specification Boolean literals must be booleans")
            elif constructor in (Q, UQ):
                if any(type(bit) is not int or bit < 0 for bit in kwargs.values()):
                    raise ValueError("Specification bit widths must be nonnegative integers")
            elif constructor is ast._CaseEntry:
                if not isinstance(kwargs["condition"], ast.BoolExpr) or not isinstance(kwargs["value"], ast.SpecNode):
                    raise ValueError("Invalid specification case entry")
            elif constructor is ast._CasePartition:
                if (type(kwargs["entries"]) is not list
                        or any(not isinstance(entry, ast._CaseEntry) for entry in kwargs["entries"])
                        or not isinstance(kwargs["value"], ast.SpecNode)):
                    raise ValueError("Invalid specification case partition")
                kwargs["entries"] = tuple(kwargs["entries"])
            result = constructor(**kwargs)
            # Some AST constructors rewrite nodes (e.g. Boolean If). Such a
            # payload is not the canonical encoding of the claimed node.
            if type(result) is not constructor:
                raise ValueError("Specification node constructor changed its type")
            return result
        raise ValueError("Unsupported specification encoding")

    try:
        return decode(value)
    except (TypeError, ZeroDivisionError, RecursionError) as exc:
        raise ValueError("Malformed specification encoding") from exc
