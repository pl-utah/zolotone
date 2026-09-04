import inspect
import typing as tp
from contextvars import ContextVar

from ..types import DataType, RuntimeValue


def _is_dtype_annotation(annotation: tp.Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, DataType)


class Node:
    _eval_cache: ContextVar[tp.Optional[dict["Node", RuntimeValue]]] = ContextVar(
        "eval_cache", default=None
    )

    def __init__(
        self,
        spec: tp.Callable[..., tp.Any],
        impl: tp.Callable[..., RuntimeValue],
        sign: tp.Callable[..., DataType],
        args: list["Node"],
        name: str,
    ):
        if not all(isinstance(arg, Node) for arg in args):
            bad_args = [type(arg).__name__ for arg in args if not isinstance(arg, Node)]
            raise TypeError(f"Node arguments must be Node instances, got {bad_args}")
        self._primitive_signature_check(sign)

        self.spec = spec
        self.sign = sign
        self.args = list(args)
        self.name = name
        self.constant: RuntimeValue | None = None
        self._fingerprint_cache: dict[bool, tp.Any] = {}

        def compute(inputs: list[RuntimeValue]) -> RuntimeValue:
            output = impl(*inputs)
            self._dynamic_typecheck(inputs, output)
            return output

        self.impl = compute
        self._static_typecheck()

    def _evaluate_spec(self, ctx, cache):
        if self in cache:
            return cache[self]
        inputs = [ctx.spec_of(arg) for arg in self.args]
        output = self.spec(*inputs, ctx=ctx)
        cache[self] = output
        return output

    def _static_typecheck(self) -> DataType:
        self.args_types = [arg.dtype for arg in self.args]
        dtype = self.sign(*self.args_types)
        if not isinstance(dtype, DataType):
            raise TypeError(
                f"Sign function for {self.name} returned non-DataType "
                f"{type(dtype).__name__}"
            )
        self.dtype = dtype
        self._signature_match(self.args_types, self.dtype)

        constants = [arg.constant for arg in self.args]
        if constants and all(value is not None for value in constants):
            self.constant = self.impl(tp.cast(list[RuntimeValue], constants))
        return self.dtype

    def _dynamic_typecheck(
        self, inputs: list[RuntimeValue], output: RuntimeValue
    ) -> None:
        if len(inputs) != len(self.args_types):
            raise TypeError(
                f"Arguments do not match Node's signature at {self.name}:\n"
                f"  Given count: {len(inputs)}\n"
                f"  Required count: {len(self.args_types)}\n"
            )
        if not all(isinstance(value, RuntimeValue) for value in inputs):
            raise TypeError(
                f"Arguments to {self.name} must be RuntimeValue instances, got "
                f"{[type(value).__name__ for value in inputs]}"
            )
        given = [value.dtype for value in inputs]
        if given != self.args_types:
            raise TypeError(
                f"Arguments do not match Node's signature at {self.name}:\n"
                f"  Given: {given}\n"
                f"  Required: {self.args_types}\n"
            )
        if not isinstance(output, RuntimeValue):
            raise TypeError(
                f"Output does not match Node's signature at {self.name}:\n"
                f"  impl returned non-RuntimeValue: {type(output).__name__}\n"
                f"  expected descriptor: {self.dtype}\n"
            )
        if output.dtype != self.dtype:
            raise TypeError(
                f"Output does not match Node's signature at {self.name}:\n"
                f"  impl: {output}\n"
                f"  impl descriptor: {output.dtype}\n"
                f"  expected descriptor: {self.dtype}\n"
            )

    def _primitive_signature_check(self, sign) -> None:
        signature = inspect.signature(sign)
        message = (
            "Signature contains types that are not DataType subclasses!\n"
            f"Given: {signature}\n"
        )
        for parameter in signature.parameters.values():
            if not _is_dtype_annotation(parameter.annotation):
                raise TypeError(message)
        if not _is_dtype_annotation(signature.return_annotation):
            raise TypeError(message)

    def _signature_match(self, args: list[DataType], output: DataType) -> None:
        signature = inspect.signature(self.sign)
        if len(args) != len(signature.parameters):
            raise TypeError(
                f"Arguments to {self.name} do not match its signature\n"
                f"Given count: {len(args)}\n"
                f"Required count: {len(signature.parameters)}\n"
            )
        required = [parameter.annotation for parameter in signature.parameters.values()]
        if any(not isinstance(arg, annotation) for arg, annotation in zip(args, required)):
            raise TypeError(
                f"Arguments to {self.name} do not match its signature\n"
                f"Given: {args}\nRequired: {required}\n"
            )
        if not isinstance(output, signature.return_annotation):
            raise TypeError(
                f"Output from {self.name} does not match its signature\n"
                f"Given: {output}\nRequired: {signature.return_annotation}"
            )

    def _fingerprint(self, jittable: bool = False):
        raise NotImplementedError

    def _cached_fingerprint(
        self,
        jittable: bool = False,
        build: tp.Callable[[], tp.Any] | None = None,
    ):
        if jittable in self._fingerprint_cache:
            return self._fingerprint_cache[jittable]
        fingerprint = build()
        self._fingerprint_cache[jittable] = fingerprint
        return fingerprint

    def _type_and_constant_fingerprint(self):
        return (
            self.dtype._fingerprint(),
            None if self.constant is None else self.constant._fingerprint(),
        )

    def to_cpp(self, name=None, jittable: bool = True):
        from ..codegen import lower_to_cpp
        return lower_to_cpp(self, name, jittable=jittable) if name else lower_to_cpp(
            self, jittable=jittable
        )

    def copy(self):
        from .helpers import Copy
        return Copy(self)

    def __getitem__(self, index: int):
        from .helpers import Tuple_get_item
        return Tuple_get_item(self, index)

    def evaluate(
        self, cache: tp.Optional[dict["Node", RuntimeValue]] = None
    ) -> RuntimeValue:
        if self.constant is not None:
            return self.constant

        active_cache = cache if cache is not None else self._eval_cache.get()
        if active_cache is None:
            active_cache = {}
        token = self._eval_cache.set(active_cache)
        try:
            if self in active_cache:
                return active_cache[self]
            inputs = [arg.evaluate(active_cache) for arg in self.args]
            output = self.impl(inputs)
            active_cache[self] = output
            return output
        finally:
            self._eval_cache.reset(token)

    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        raise NotImplementedError
