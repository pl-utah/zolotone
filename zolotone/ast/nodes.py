import inspect
import random
import typing as tp

from ..types import DataType, Value
from ..types.tuple import Tuple, TuplePattern
from ..utils import make_fixed_arguments
from ..solver.report import (
    CheckResult,
    StdoutVerificationObserver,
    VerificationObserver,
)
from .case_split import run_equivalence_cases
from .node import Node
from .parallel_verification import resolve_max_workers
from .proofs import SpecRecorder, record_specs
from ..spec import SpecContext, SpecNode, special_encoding


CLowering = tp.Callable[[list[str], bool], str]


class _SpecContract(tp.NamedTuple):
    signature: inspect.Signature
    annotations: dict[str, DataType | type[DataType] | TuplePattern]
    display_name: str


def _spec_display_name(name: str, spec: tp.Callable[..., tp.Any]) -> str:
    function_name = getattr(
        spec,
        "__qualname__",
        getattr(spec, "__name__", repr(spec)),
    )
    return f"{name!r} ({function_name})"


def _build_spec_contract(
    name: str,
    spec: tp.Callable[..., tp.Any],
) -> _SpecContract:
    """Resolve and validate a Primitive/Composite dtype contract."""

    display_name = _spec_display_name(name, spec)
    try:
        signature = inspect.signature(spec)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Specification {display_name} has no inspectable signature: {exc}"
        ) from exc

    parameters = list(signature.parameters.values())
    if not parameters:
        raise TypeError(
            f"Specification {display_name} is missing a final SpecContext parameter"
        )

    try:
        resolved = tp.get_type_hints(spec)
    except Exception as exc:
        raise TypeError(
            f"Specification {display_name} has an annotation that could not be "
            f"resolved: {getattr(spec, '__annotations__', {})!r}"
        ) from exc

    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"Specification {display_name} uses variadic parameter "
                f"{parameter.name!r}; variadic specifications are not supported"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise TypeError(
                f"Specification {display_name} parameter {parameter.name!r} "
                "has a default value; default parameters are not supported"
            )

    ctx_param = parameters[-1]
    ctx_annotation = resolved.get(ctx_param.name, ctx_param.annotation)
    if (
        ctx_param.kind is inspect.Parameter.KEYWORD_ONLY
        or (
            ctx_annotation is not inspect.Parameter.empty
            and ctx_annotation is not SpecContext
        )
    ):
        raise TypeError(
            f"Specification {display_name} is missing a final SpecContext "
            f"parameter; last parameter {ctx_param.name!r} has "
            f"annotation {ctx_annotation!r}"
        )

    contract_annotations = {}
    for parameter in parameters[:-1]:
        if parameter.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"Specification {display_name} is missing an annotation for "
                f"parameter {parameter.name!r}"
            )
        annotation = resolved.get(parameter.name, parameter.annotation)
        contract_annotations[parameter.name] = annotation

    if signature.return_annotation is inspect.Signature.empty:
        raise TypeError(
            f"Specification {display_name} is missing a return annotation"
        )
    contract_annotations["return"] = resolved.get(
        "return", signature.return_annotation
    )

    for annotation_name, annotation in contract_annotations.items():
        _validate_contract_annotation(
            display_name,
            annotation_name,
            annotation,
        )

    return _SpecContract(
        signature=signature,
        annotations=contract_annotations,
        display_name=display_name,
    )


def _validate_contract_annotation(
    display_name: str,
    annotation_name: str,
    annotation: object,
) -> None:
    if isinstance(annotation, TuplePattern):
        for index, item_annotation in enumerate(annotation.items):
            _validate_contract_annotation(
                display_name,
                f"{annotation_name}[{index}]",
                item_annotation,
            )
        return
    if annotation is Tuple:
        raise TypeError(
            f"Specification {display_name} has incomplete annotation for "
            f"{annotation_name!r}: bare Tuple does not specify its item types"
        )
    if isinstance(annotation, DataType):
        return
    if isinstance(annotation, type) and issubclass(annotation, DataType):
        return
    raise TypeError(
        f"Specification {display_name} has invalid annotation for "
        f"{annotation_name!r}: {annotation!r}; expected a DataType descriptor, "
        "DataType subclass, or populated Tuple contract"
    )


def _annotation_matches(
    dtype: DataType,
    annotation: DataType | type[DataType] | TuplePattern,
) -> bool:
    if isinstance(annotation, TuplePattern):
        return (
            isinstance(dtype, Tuple)
            and len(dtype.items) == len(annotation.items)
            and all(
                _annotation_matches(item_dtype, item_annotation)
                for item_dtype, item_annotation in zip(
                    dtype.items,
                    annotation.items,
                )
            )
        )
    if isinstance(annotation, DataType):
        return dtype == annotation
    return isinstance(dtype, annotation)


def _validate_spec_inputs(
    contract: _SpecContract,
    args: tuple[Node, ...],
) -> None:
    if not all(isinstance(arg, Node) for arg in args):
        bad_args = [type(arg).__name__ for arg in args if not isinstance(arg, Node)]
        raise TypeError(
            f"Arguments to specification {contract.display_name} must be Node "
            f"instances, got {bad_args}"
        )

    marker = object()
    try:
        bound = contract.signature.bind(*([marker] * (len(args) + 1)))
    except TypeError as exc:
        raise TypeError(
            f"Arguments do not match specification {contract.display_name}: {exc}"
        ) from exc

    arg_index = 0
    for parameter in list(contract.signature.parameters.values())[:-1]:
        annotation = contract.annotations[parameter.name]
        if parameter.name in bound.arguments:
            dtype = args[arg_index].dtype
            if not _annotation_matches(dtype, annotation):
                raise TypeError(
                    f"Input {arg_index} to specification {contract.display_name} "
                    f"has descriptor {dtype!r}; expected {annotation!r}"
                )
            arg_index += 1


def _validate_spec_output(contract: _SpecContract, inner_tree: Node) -> None:
    if not isinstance(inner_tree, Node):
        raise TypeError(
            f"Implementation for specification {contract.display_name} returned "
            f"{type(inner_tree).__name__}, expected Node"
        )
    annotation = contract.annotations["return"]
    if not _annotation_matches(inner_tree.dtype, annotation):
        raise TypeError(
            f"Output from specification {contract.display_name} has descriptor "
            f"{inner_tree.dtype!r}; expected {annotation!r}"
        )


class _Spec(tp.NamedTuple):
    name: str
    collect: tp.Callable[[SpecContext], tp.Any]
    partition_cases: bool = False


def _default_equivalence_schedule() -> list[dict[str, tp.Any]]:
    schedule = []
    for _ in range(3):
        schedule.extend(
            [
                {"tool": "simplify"},
                {
                    "tool": "egglog-rewrite",
                    "iterations": 7,
                    "scheduler": {
                        "match_limit": 500_000,
                        "ban_length": 1,
                    },
                },
            ]
        )
    schedule.append({"tool": "z3", "timeout_ms": 10000})
    return schedule


def check_equivalence(
    first: _Spec,
    second: _Spec,
    base_ctx: SpecContext,
    inputs: list[tp.Any],
    schedule: list[str | dict[str, tp.Any]] | None = None,
    observer: VerificationObserver | None = None,
    max_workers: int | None = None,
):
    worker_count = resolve_max_workers(max_workers)
    if first.name == second.name:
        raise ValueError("Equivalent specification sides must have distinct names")
    if schedule is None:
        schedule = _default_equivalence_schedule()
    active_observer = (
        observer if observer is not None else StdoutVerificationObserver()
    )

    base_assume_count = len(base_ctx.assumes)
    base_check_count = len(base_ctx.checks)
    base_requirement_count = len(base_ctx.requirements)

    first_ctx = base_ctx.copy()
    first_output = first.collect(first_ctx)

    second_ctx = base_ctx.copy()
    # Collection is isolated, while fresh names remain globally distinct.
    second_ctx._sym_counter = first_ctx._sym_counter
    second_output = second.collect(second_ctx)

    combined_ctx = base_ctx.copy(
        assumes=list(base_ctx.assumes) + first_ctx.assumes[base_assume_count:] + second_ctx.assumes[base_assume_count:],
        checks=list(base_ctx.checks) + first_ctx.checks[base_check_count:] + second_ctx.checks[base_check_count:],
        requirements=list(base_ctx.requirements) + first_ctx.requirements[base_requirement_count:] + second_ctx.requirements[base_requirement_count:],
    )
    combined_ctx._sym_counter = second_ctx._sym_counter

    requirement_report = combined_ctx.validate_requirements()

    preferred_sides = [
        side_index
        for side_index, spec in enumerate((first, second))
        if spec.partition_cases
    ]
    case_results = run_equivalence_cases(
        combined_ctx=combined_ctx,
        side_contexts=(first_ctx, second_ctx),
        outputs=(first_output, second_output),
        inputs=inputs,
        schedule=schedule,
        observer=active_observer,
        max_workers=worker_count,
        preferred_side=preferred_sides[0] if preferred_sides else None,
    )
    proved = all(result["proved"] for result in case_results)

    return CheckResult(
        proved=proved,
        requirement_report=requirement_report,
        cases=case_results,
    )


def _check_determinism(
    node: "composite | primitive",
    schedule: list[str | dict[str, tp.Any]] | None = None,
    observer: VerificationObserver | None = None,
    max_workers: int | None = None,
):
    base_ctx = SpecContext(f"{node.name}_determinism")
    inputs = [base_ctx.spec_of(arg) for arg in node.inner_args]

    def collect_spec(ctx):
        encoded_inputs = [special_encoding(value, ctx) for value in inputs]
        return node.spec(*encoded_inputs, ctx)

    first_spec = _Spec("first_spec", collect_spec, partition_cases=True)
    second_spec = _Spec("second_spec", collect_spec)

    result = check_equivalence(
        first_spec,
        second_spec,
        base_ctx=base_ctx,
        inputs=inputs,
        schedule=schedule,
        observer=observer,
        max_workers=max_workers,
    )

    print(f"{node.name} specification {'is' if result['proved'] else 'is not'} deterministic")

    return result


def Composite(
    name: str,
    spec: tp.Callable[..., tp.Any],
    c_inline: bool = False,
    c_lowering: tp.Optional[CLowering] = None,
):
    contract = _build_spec_contract(name, spec)

    def wrapper1(impl: tp.Callable[..., Node]):
        def wrapper2(*args):
            return composite(
                spec=spec,
                spec_contract=contract,
                impl=impl,
                args=args,
                name=name,
                c_inline=c_inline,
                c_lowering=c_lowering,
            )
        setattr(wrapper2, "_spec_contract", contract)
        return wrapper2
    return wrapper1


class composite(Node):
    def __init__(
        self,
        spec: tp.Callable[..., tp.Any],
        spec_contract: _SpecContract,
        impl: tp.Callable[..., Node],
        args: list[Node],
        name: str,
        c_inline: bool = False,
        c_lowering: tp.Optional[CLowering] = None,
    ):
        self.c_inline = c_inline
        self.c_lowering = c_lowering
        self.ctx = SpecContext(name)
        self.spec_assumes: tuple[SpecNode, ...] = ()
        self.spec_checks: tuple[SpecNode, ...] = ()
        self.inner_assumes: tuple[Node, ...] = ()
        self.inner_checks: tuple[Node, ...] = ()
        _validate_spec_inputs(spec_contract, args)
        self.inner_args = [
            Var(name=f"arg_{i}", dtype=x.dtype, constant=x.constant)
            for i, x in enumerate(args)
        ]
        
        recorder = SpecRecorder(self.ctx)
        with record_specs(recorder):
            self.inner_tree = impl(*self.inner_args)

        _validate_spec_output(spec_contract, self.inner_tree)
        
        self._validate_components(name)
        
        def impl_(*args):
            for var, arg in zip(self.inner_args, args):
                var.load_value(Value(var.dtype, arg))
            return self.inner_tree.evaluate().raw
        
        # Signature is obtained from the inner tree
        def sign(*args):
            return self.inner_tree.dtype
        
        sign = make_fixed_arguments(
            sign,
            arg_types=[type(x.dtype) for x in args],
            return_type=type(self.inner_tree.dtype),
        )
        
        super().__init__(
            spec=spec,
            impl=impl_,
            sign=sign,
            args=args,
            name=name,
        )
    
    def check_spec(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        base_ctx = self.ctx.copy()
        inputs = [base_ctx.spec_of(arg) for arg in self.inner_args]

        def collect_inner(ctx):
            encoded_inputs = [
                special_encoding(value, ctx)
                for value in inputs
            ]
            for node, value in zip(self.inner_args, encoded_inputs):
                ctx.spec_cache[node] = value
            return ctx.spec_of(self.inner_tree)

        def collect_outer(ctx):
            encoded_inputs = [
                special_encoding(value, ctx)
                for value in inputs
            ]
            return self.spec(*encoded_inputs, ctx)

        result = check_equivalence(
            _Spec("inner_spec", collect_inner),
            _Spec("outer_spec", collect_outer, partition_cases=True),
            base_ctx=base_ctx,
            inputs=inputs,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )
        
        print(f"{self.ctx.name} {'has' if result['proved'] else 'has not'} been proved")
        
        return result

    def check_determinism(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        return _check_determinism(
            self,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )

    def set_impl_conditions(
        self,
        *,
        assumes: tp.Iterable[tuple[SpecNode, Node]],
        checks: tp.Iterable[tuple[SpecNode, Node]],
    ) -> None:
        """Attach lowered Boolean preconditions and postconditions."""
        assume_pairs = tuple(assumes)
        check_pairs = tuple(checks)
        self.spec_assumes = tuple(spec for spec, _ in assume_pairs)
        self.spec_checks = tuple(spec for spec, _ in check_pairs)
        self.inner_assumes = tuple(node for _, node in assume_pairs)
        self.inner_checks = tuple(node for _, node in check_pairs)
        self._validate_components(self.name)
        self._fingerprint_cache.clear()
    
    def _validate_components(self, composite_name: str) -> None:
        visited: set[Node] = set()
        
        def visit(node: Node, path: str) -> None:
            if node in visited:
                return
            visited.add(node)
            
            if isinstance(node, (primitive, composite)):
                for idx, arg in enumerate(node.args):
                    visit(arg, f"{path} -> {node.name}.arg[{idx}]")
                return
            
            if isinstance(node, (Var, Const)):
                return
            
            raise TypeError(
                f"Composite {composite_name} must be composed recursively of Primitive/Composite nodes; "
                f"found {type(node).__name__} {node.name!r} at {path}"
            )
        
        visit(self.inner_tree, f"{composite_name}.impl")
        for kind, conditions in (
            ("assume", self.inner_assumes),
            ("check", self.inner_checks),
        ):
            for index, condition in enumerate(conditions):
                visit(condition, f"{composite_name}.{kind}[{index}]")
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.dtype}: {self.name} [Composite]")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        if depth > 0:
            print(new_prefix + "└── Impl:")
            self.inner_tree.print_tree(new_prefix + "    ", True, depth - 1)
        else:
            for i, arg in enumerate(self.args):
                is_arg_last = i == len(self.args) - 1
                arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"[Composite] {self.name}: {' -> '.join([str(x) for x in self.args_types])} -> {self.dtype}"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            direct_cpp_lowering = None
            if self.c_lowering is not None:
                direct_cpp_lowering = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                type(self).__name__,
                self.name,
                self._type_and_constant_fingerprint(),
                tuple(arg._type_and_constant_fingerprint() for arg in self.inner_args),
                direct_cpp_lowering,
                self.inner_tree._fingerprint(jittable) if direct_cpp_lowering is None else None,
                tuple(
                    (str(spec), assumption._fingerprint(jittable))
                    for spec, assumption in zip(
                        self.spec_assumes,
                        self.inner_assumes,
                        strict=True,
                    )
                ),
                tuple(
                    (str(spec), check._fingerprint(jittable))
                    for spec, check in zip(
                        self.spec_checks,
                        self.inner_checks,
                        strict=True,
                    )
                ),
            )
        
        return self._cached_fingerprint(jittable, build)


def Primitive(
    name: str,
    spec: tp.Callable[..., tp.Any],
    c_inline: bool = False,
    c_lowering: tp.Optional[CLowering] = None,
):
    contract = _build_spec_contract(name, spec)

    def wrapper1(impl: tp.Callable[..., Node]):
        def wrapper2(*args):
            return primitive(
                spec=spec,
                spec_contract=contract,
                impl=impl,
                args=args,
                name=name,
                c_inline=c_inline,
                c_lowering=c_lowering,
            )
        setattr(wrapper2, "_spec_contract", contract)
        return wrapper2
    return wrapper1

class primitive(Node):
    def __init__(
        self,
        spec: tp.Callable[..., tp.Any],
        spec_contract: _SpecContract,
        impl: tp.Callable[..., Node],
        args: list[Node],
        name: str,
        c_inline: bool = False,
        c_lowering: tp.Optional[CLowering] = None,
    ):
        self.c_inline = c_inline
        self.c_lowering = c_lowering
        _validate_spec_inputs(spec_contract, args)
        self.inner_args = [
            Var(name=f"arg_{i}", dtype=x.dtype, constant=x.constant)
            for i, x in enumerate(args)
        ]
        
        self.inner_tree = impl(*self.inner_args)

        _validate_spec_output(spec_contract, self.inner_tree)
        
        def impl_(*args):
            for var, arg in zip(self.inner_args, args):
                if isinstance(var, Var):
                    var.load_value(Value(var.dtype, arg))
            return self.inner_tree.evaluate().raw
        
        # Signature is obtained from the inner tree
        def sign(*args):
            return self.inner_tree.dtype
        
        sign = make_fixed_arguments(
            sign,
            arg_types=[type(x.dtype) for x in args],
            return_type=type(self.inner_tree.dtype),
        )
        
        super().__init__(
            spec=spec,
            impl=impl_,
            sign=sign,
            args=args,
            name=name,
        )

    def check_determinism(
        self,
        schedule: list[str | dict[str, tp.Any]] | None = None,
        observer: VerificationObserver | None = None,
        max_workers: int | None = None,
    ):
        return _check_determinism(
            self,
            schedule=schedule,
            observer=observer,
            max_workers=max_workers,
        )
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.dtype}: {self.name} [Primitive]")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        if depth > 0:
            print(new_prefix + "└── Impl:")
            self.inner_tree.print_tree(new_prefix + "    ", True, depth - 1)
        else:
            for i, arg in enumerate(self.args):
                is_arg_last = i == len(self.args) - 1
                arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"[Primitive] {self.name}: {' -> '.join([str(x) for x in self.args_types])} -> {self.dtype}"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            direct_cpp_lowering = None
            if self.c_lowering is not None:
                direct_cpp_lowering = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                type(self).__name__,
                self.name,
                self._type_and_constant_fingerprint(),
                tuple(arg._type_and_constant_fingerprint() for arg in self.inner_args),
                direct_cpp_lowering,
                self.inner_tree._fingerprint(jittable) if direct_cpp_lowering is None else None,
            )
        
        return self._cached_fingerprint(jittable, build)


class Op(Node):
    def __init__(
        self,
        impl: tp.Callable[..., object],
        sign: tp.Callable[..., DataType],
        args: list[Node],
        name: str,
        c_lowering: tp.Optional[CLowering],
    ):
        self.c_lowering = c_lowering
        super().__init__(
            spec=None,
            impl=impl,
            sign=sign,
            args=args,
            name=name,
        )
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.__str__())
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, arg in enumerate(self.args):
            is_arg_last = i == len(self.args) - 1
            arg.print_tree(new_prefix, is_arg_last, depth)
    
    def __str__(self):
        return f"{self.dtype}: {self.name} [Op]"
    
    def _fingerprint(self, jittable: bool = False):
        def build():
            lowering_fingerprint = None
            if self.c_lowering is not None:
                lowering_fingerprint = self.c_lowering(
                    [f"${idx}" for idx in range(len(self.args))],
                    jittable,
                )
            return (
                "Op",
                self.name,
                self._type_and_constant_fingerprint(),
                lowering_fingerprint,
                tuple(arg._fingerprint(jittable) for arg in self.args),
            )
        
        return self._cached_fingerprint(jittable, build)


class Const(Node):
    def __init__(self, value: Value):
        if not isinstance(value, Value):
            raise TypeError(
                f"Const value must be a Value, got {type(value).__name__}"
            )
        self.value = value
        
        def impl():
            return self.value.raw
        
        def spec(ctx):
            return self.value.to_spec(ctx)
        
        def sign():
            return self.value.dtype

        sign = make_fixed_arguments(
            sign, arg_types=[], return_type=type(value.dtype)
        )
        
        super().__init__(
            spec=spec,
            impl=impl,
            sign=sign,
            args=[],
            name=str(self.value.to_python()),
        )
        self.constant = self.value
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + self.__str__())
    
    def __str__(self):
        return f"{self.dtype}: {self.name if self.name else str(self.value)} [Const]"
    
    def _fingerprint(self, jittable: bool = False):
        return self._cached_fingerprint(
            jittable,
            lambda: ("Const", self.constant._fingerprint()),
        )


class Var(Node):
    def __init__(
        self,
        name: str,
        dtype: DataType,
        *,
        constant: Value | None = None,
    ):
        if not isinstance(dtype, DataType):
            raise TypeError(f"Var dtype must be a DataType, got {type(dtype).__name__}")
        if constant is not None:
            if not isinstance(constant, Value):
                raise TypeError(
                    f"Var constant must be a Value, got {type(constant).__name__}"
                )
            if constant.dtype != dtype:
                raise TypeError(
                    f"Var constant descriptor {constant.dtype} does not match {dtype}"
                )
        self._value = None
        
        def impl():
            if self._value is None:
                raise ValueError(f"Variable {self.name} not bound to a value")
            return self._value.raw
        
        def spec(ctx):
            return dtype.to_spec(self.name, ctx)
        
        def signature():
            return dtype

        signature = make_fixed_arguments(
            signature, arg_types=[], return_type=type(dtype)
        )
        
        super().__init__(
            spec=spec,
            impl=impl,
            sign=signature,
            args=[],
            name=name,
        )
        self.constant = constant
    
    def load_rand(self, rng: tp.Optional[random.Random] = None):
        if rng is None:
            rng = random.Random()
        self.load_value(self.dtype.random_value(rng))
    
    def print_tree(self, prefix: str = "", is_last: bool = True, depth: int = 0):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + f"{self.dtype}: {self.name} [Var]")
    
    def load_value(self, value: Value):
        if not isinstance(value, Value):
            raise TypeError(
                f"Var value must be a Value, got {type(value).__name__}"
            )
        if value.dtype != self.dtype:
            raise TypeError(
                f"Var value descriptor does not match {self.dtype}; got {value.dtype}"
            )
        self._value = value
    
    def __str__(self):
        return f"{self.dtype}: {self.name} [Var]"
    
    def _fingerprint(self, jittable: bool = False):
        return self._cached_fingerprint(
            jittable,
            lambda: (
                "Var",
                self.name,
                self.dtype._fingerprint(),
                None if self.constant is None else self.constant._fingerprint(),
            ),
        )
    
