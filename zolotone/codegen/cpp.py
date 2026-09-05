from __future__ import annotations

from dataclasses import dataclass, field
import typing as tp

from ..ast.node import Node
from ..ast.nodes import CLowering, Const, Op, Var, composite, primitive
from ..types import DataType, RuntimeValue, Tuple, TupleValue


class CppLoweringError(RuntimeError):
    pass


@dataclass(frozen=True)
class _CppValue:
    expr: str
    tuple_items: tuple["_CppValue", ...] | None = None


# Helper object for lowering one function at a time
@dataclass
class _FunctionContext:
    memo: dict[Node, _CppValue] = field(default_factory=dict)
    statements: list[str] = field(default_factory=list)


class _CppEmitter:
    def __init__(self, jittable: bool = True) -> None:
        self.jittable = jittable
        self._reserved_names = {}
        self._function_cache: dict[tp.Any, str] = {}
        self._functions: list[str] = []

    def emit_cpp(self, root: Node, function_name: str) -> str:
        if isinstance(root.dtype, Tuple):
            raise CppLoweringError("Tuple-typed entry points are not supported in C++ lowering")
        public_name = self._make_name(function_name)
        internal_name = self._make_name(f"{public_name}_impl")
        self.emit_function(root=root, function_name=internal_name)
        self._functions.append(
            self._render_public_wrapper(
                public_name=public_name,
                internal_name=internal_name,
                args=root.inner_args,
                return_type=root.dtype,
            )
        )
        includes = ["#include <cstdint>"]
        if self.jittable:
            includes.append("#include <array>")
        else:
            includes.extend([
                "#include <tuple>",
                "#include <ac_int.h>",
            ])
        parts = [
            "#pragma once",
            *includes,
            "",
        ]
        if not self.jittable:
            parts.extend([
                "template <int W>",
                "using ac_uint = ac_int<W, false>;",
                "",
            ])
        parts.extend(self._functions)
        return "\n".join(parts)

    def emit_function(self, root: Node, function_name: str) -> str:
        assert isinstance(root, (composite, primitive)), "Can lower only Primitive/Composite"

        cache_key = root._fingerprint(self.jittable)
        if cache_key in self._function_cache:
            return self._function_cache[cache_key]
        self._function_cache[cache_key] = function_name

        env = {
            arg: _CppValue(expr=arg.name)
            for arg in root.inner_args
        }

        rendered = self._render_function(
            root=root,
            name=function_name,
            env=env,
        )
        self._functions.append(rendered)
        return function_name

    def _render_function(
        self,
        root: composite | primitive,
        name: str,
        env: dict[Node, _CppValue],
    ) -> str:
        ctx = _FunctionContext()
        if root.c_lowering is not None:
            result = self._lower_direct_cpp(
                root.dtype,
                root.c_lowering,
                [env[arg].expr for arg in root.inner_args],
            )
        else:
            result = self._lower(root.inner_tree, env, ctx)

        signature = f"static inline {self._signature(name, root.inner_args, root.dtype)}"

        body = [*ctx.statements, f"return {result.expr};"]
        indented_body = "\n".join(f"    {line}" for line in body)
        return "\n".join([signature + f" {{  // {root.name}", indented_body, "}"])

    def _should_inline(self, node: Node) -> bool:
        return isinstance(node, (primitive, composite)) and (
            node.c_inline
            or isinstance(node.dtype, Tuple)  # FOR NOW. SOME TROUBLES WIT JIT AND ARRAYS - JUST INLINE EVERY TUPLE
        )

    def _lower(
        self,
        node: Node,
        env: dict[Node, _CppValue],
        ctx: _FunctionContext,
    ) -> _CppValue:
        if isinstance(node, Var):
            try:
                return env[node]
            except KeyError as exc:
                raise CppLoweringError(
                    f"Unbound variable during lowering: {node.name}"
                ) from exc

        if node in ctx.memo:
            return ctx.memo[node]

        if node.constant is not None:
            lowered = self._lower_const(node.constant)
            ctx.memo[node] = lowered
            return lowered

        if isinstance(node, Const):
            lowered = self._lower_const(node.value)
            ctx.memo[node] = lowered
            return lowered

        if isinstance(node, Op):
            lowered = self._lower_op(node, env, ctx)
            ctx.memo[node] = lowered
            return lowered

        if isinstance(node, (primitive, composite)):
            lowered_args = [self._lower(arg, env, ctx) for arg in node.args]
            if self._should_inline(node):  # Inlining functions into current call
                if not node.c_inline:
                    ctx.statements.append(f"// begin inline {type(node).__name__} {node.name}")
                    
                if node.c_lowering is not None:
                    lowered = self._lower_direct_cpp(
                        node.dtype,
                        node.c_lowering,
                        [arg.expr for arg in lowered_args],
                    )
                else:
                    inline_env = dict(env)
                    inline_env.update(dict(zip(node.inner_args, lowered_args)))
                    lowered = self._lower(node.inner_tree, inline_env, ctx)

                if not node.c_inline:
                    ctx.statements.append(f"// end inline {type(node).__name__} {node.name}")
                ctx.memo[node] = lowered
                return lowered
            else:  # Create a separate function for the node
                helper_name = self._function_cache.get(node._fingerprint(self.jittable))
                if helper_name is None:
                    helper_name = self._make_name(node.name)
                    self.emit_function(
                        root=node,
                        function_name=helper_name,
                    )
                expr = f"{helper_name}({', '.join(arg.expr for arg in lowered_args)})"
                lowered = self._emit_temp(node.dtype, expr, node.name, ctx)
                ctx.memo[node] = lowered
                return lowered

        raise CppLoweringError(f"Unsupported node type: {type(node).__name__}")
    def _lower_const(self, value: RuntimeValue) -> _CppValue:
        tuple_items = None
        if isinstance(value, TupleValue):
            tuple_items = tuple(self._lower_const(item) for item in value.items)
        return _CppValue(
            expr=self._const_expr(value),
            tuple_items=tuple_items,
        )

    def _const_expr(self, value: RuntimeValue) -> str:
        if isinstance(value, TupleValue):
            for item in value.items:
                if isinstance(item, TupleValue):
                    raise CppLoweringError("Nested tuples are not supported in C++ lowering")
            args = [self._lower_const(item) for item in value.items]
            if self.jittable:
                return (
                    f"{self._render_type(value.dtype)}{{"
                    + ", ".join(f"static_cast<uint64_t>({arg.expr})" for arg in args)
                    + "}"
                )
            return f"std::make_tuple({', '.join(arg.expr for arg in args)})"
        return self._cast(value.dtype, str(value.raw))
    
    def _lower_op(
        self,
        node: Op,
        env: dict[Node, _CppValue],
        ctx: _FunctionContext,
    ) -> _CppValue:
        if node.c_lowering is None:
            raise CppLoweringError(f"Unsupported op lowering for {node.name}")

        lowered_args = [self._lower(arg, env, ctx) for arg in node.args]

        # Skipping tuple creation
        if node.name.startswith("_basic_get_item_"):
            source = lowered_args[0]
            if source.tuple_items is not None:
                idx = int(node.name.rsplit("_", 1)[1])
                return source.tuple_items[idx]

        lowered_arg_exprs = [arg.expr for arg in lowered_args]
        expr = self._cast(node.dtype, node.c_lowering(lowered_arg_exprs, self.jittable))

        # Skipping tuple creation
        if node.name.startswith("basic_tuple_maker_"):
            return _CppValue(
                expr=expr,
                tuple_items=tuple(lowered_args),
            )

        return self._emit_temp(node.dtype, expr, node.name, ctx)
    
    def _signature(self, name: str, args: list[Var], return_type: DataType) -> str:
        params_sig = ", ".join(
            f"{self._render_type(arg.dtype)} {arg.name}" for arg in args
        )
        return f"{self._render_type(return_type)} {name}({params_sig})"

    def _render_public_wrapper(
        self,
        public_name: str,
        internal_name: str,
        args: list[Var],
        return_type: DataType,
    ) -> str:
        call_args = ", ".join(self._render_public_arg(arg) for arg in args)
        wrapper_signature = self._signature(
            name=public_name,
            args=args,
            return_type=return_type,
        )
        return "\n".join(
            [
                f'extern "C" inline {wrapper_signature} {{',
                f"    return {internal_name}({call_args});",
                "}",
            ]
        )
    
    def _render_type(self, type_: DataType) -> str:
        if isinstance(type_, Tuple) and any(isinstance(item, Tuple) for item in type_.items):
            raise CppLoweringError("Nested tuples are not supported in C++ lowering")
        return type_.to_cpp_type(jittable=self.jittable)

    def _lower_direct_cpp(
        self,
        return_type: DataType,
        c_lowering: CLowering,
        arg_exprs: list[str],
    ) -> _CppValue:
        if isinstance(return_type, Tuple):
            raise CppLoweringError("Custom C++ lowering does not support tuple outputs")
        expr = self._cast(return_type, c_lowering(arg_exprs, self.jittable))
        return _CppValue(expr=expr)
    
    def _cast(self, type_: DataType, expr: str) -> str:
        if isinstance(type_, Tuple):
            return expr
        if self.jittable:
            return f"{self._render_type(type_)}({self._mask(expr, type_)})"
        return f"{self._render_type(type_)}({expr})"

    def _mask(self, expr: str, type_):
        return f"({expr}) & {self._mask_literal(type_.total_bits())}"

    def _mask_literal(self, bits: int) -> str:
        return str((1 << bits) - 1)

    def _render_public_arg(self, arg: Var) -> str:
        if not self.jittable:
            return arg.name
        return f"({arg.name} & {self._mask_literal(arg.dtype.total_bits())})"
    
    def _emit_temp(
        self,
        type_: DataType,
        expr: str,
        name: str,
        ctx: _FunctionContext,
    ) -> _CppValue:
        temp_name = self._make_name("tmp")
        cpp_type = self._render_type(type_)
        ctx.statements.append(f"const {cpp_type} {temp_name} = {expr};  // {name}")
        return _CppValue(expr=temp_name)
    
    def _make_name(self, base: str) -> str:
        safe_base = self._sanitize_identifier(base)
        name = safe_base
        i = 0
        while True:
            i += 1
            if name in self._reserved_names:
                name = f"{safe_base}_{i}"
            else:
                self._reserved_names[name] = True
                break
        return name
    
    def _sanitize_identifier(self, name: str) -> str:
        chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in name]
        safe = "".join(chars).strip("_") or "tmp"
        if safe[0].isdigit():
            safe = f"_{safe}"
        return safe


def lower_to_cpp(
    root: Node,
    function_name: str | None = None,
    jittable: bool = True,
) -> str:
    if function_name is None:
        function_name = root.name
    emitter = _CppEmitter(jittable=jittable)
    return emitter.emit_cpp(root=root, function_name=function_name)
