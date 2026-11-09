import shutil
import tempfile
import unittest
import ctypes
import subprocess
import re
from pathlib import Path

from ..ast.node import Node
from ..types import DataType, Tuple

_AC_TYPES_INCLUDE_DIR = Path(__file__).resolve().parents[2] / "infra" / "ac_types" / "include"


class CppAssertionError(AssertionError):
    """An assertion failed while executing generated C++."""


def _find_cpp_compiler() -> str | None:
    for candidate in ("c++", "g++", "clang++"):
        compiler = shutil.which(candidate)
        if compiler is not None:
            return compiler
    raise unittest.SkipTest("A C++ compiler is required for lowering round-trip tests")


def _abi_bits(type_: DataType) -> int:
    if isinstance(type_, Tuple):
        raise TypeError("compile helpers support only scalar arguments and return types")

    total_bits = type_.total_bits()
    if total_bits <= 8:
        return 8
    if total_bits <= 16:
        return 16
    if total_bits <= 32:
        return 32
    if total_bits <= 64:
        return 64
    raise TypeError("compile helpers support only values up to 64 bits")


def _cpp_abi_type(type_: DataType) -> str:
    return f"std::uint{_abi_bits(type_)}_t"


def _lowered_cpp_type(type_: DataType, jittable: bool) -> str:
    return type_.to_cpp_type(jittable=jittable)


def _ctypes_abi_type(type_: DataType):
    bits = _abi_bits(type_)
    if bits == 8:
        return ctypes.c_uint8
    if bits == 16:
        return ctypes.c_uint16
    if bits == 32:
        return ctypes.c_uint32
    return ctypes.c_uint64


def _wrapper_arg_expr(arg, idx: int, *, jittable: bool) -> str:
    expr = f"arg_{idx}"
    lowered_type = _lowered_cpp_type(arg.dtype, jittable=jittable)
    return f"static_cast<{lowered_type}>({expr})"


def compile_(node: Node, jittable: bool, *, catch_assertions: bool = False):
    if not hasattr(node, "inner_args"):
        raise TypeError("compile helpers expect a Primitive or Composite node")

    function_name = "lowered_entry"
    source = node.to_cpp(function_name, jittable=jittable)
    if catch_assertions:
        # Keep assertion expressions and evaluation order, but throw within the
        # C++ wrapper instead of aborting the Python process. Never let a C++
        # exception cross the ctypes boundary.
        source = (
            "struct ZfvAssertion { const char *expression; };\n"
            "#define ZFV_ASSERT(condition) "
            "((condition) ? (void)0 : throw ZfvAssertion{#condition})\n"
            + re.sub(r"\bassert\s*\(", "ZFV_ASSERT(", source)
        )
    compiler = _find_cpp_compiler()

    tempdir = tempfile.TemporaryDirectory()
    temp_path = Path(tempdir.name)
    source_path = temp_path / "lowered.cpp"
    wrapper_path = temp_path / "wrapper.cpp"
    library_path = temp_path / "lowered.so"

    source_path.write_text(source, encoding="utf-8")
    arg_decls = [
        f"{_cpp_abi_type(arg.dtype)} arg_{idx}"
        for idx, arg in enumerate(node.inner_args)
    ]
    call_args = ", ".join(
        _wrapper_arg_expr(arg, idx, jittable=jittable)
        for idx, arg in enumerate(node.inner_args)
    )
    return_type = _cpp_abi_type(node.dtype)
    body = [f"return static_cast<{return_type}>(Zolotone::zolotone({call_args}));"]
    if catch_assertions:
        arg_decls.append("const char **assertion")
        body = [
            "*assertion = nullptr;",
            "try {",
            f"    {body[0]}",
            "} catch (const ZfvAssertion &failure) {",
            "    *assertion = failure.expression;",
            "    return 0;",
            "}",
        ]

    wrapper_path.write_text(
        "\n".join(
            [
                "#include <cstdint>",
                '#include "lowered.cpp"',
                "",
                f'extern "C" {return_type} {function_name}_entry({", ".join(arg_decls)}) {{',
                *(f"    {line}" for line in body),
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    command = [
        compiler,
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O3",
        "-march=native",
        f"-I{_AC_TYPES_INCLUDE_DIR}",
        str(wrapper_path),
        "-o",
        str(library_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tempdir.cleanup()
        extra_note = ""
        if not jittable and ("ac_int.h" in result.stderr or "ac_uint.h" in result.stderr):
            extra_note = (
                "\nnonjit_compile requires the Algorithmic C ac_datatypes headers "
                "(for example, ac_int.h, which defines ac_uint) to be available to the C++ compiler.\n"
            )
        raise AssertionError(
            "Failed to compile lowered C++:\n"
            f"{extra_note}"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    library = ctypes.CDLL(str(library_path))
    func = getattr(library, f"{function_name}_entry")
    func.argtypes = [_ctypes_abi_type(arg.dtype) for arg in node.inner_args]
    func.restype = _ctypes_abi_type(node.dtype)
    if catch_assertions:
        func.argtypes = [*func.argtypes, ctypes.POINTER(ctypes.c_char_p)]

        def evaluate(*args):
            assertion = ctypes.c_char_p()
            value = func(*args, ctypes.byref(assertion))
            if assertion.value is not None:
                raise CppAssertionError(assertion.value.decode("utf-8"))
            return value

        return tempdir, evaluate
    return tempdir, func


def jit_compile(node: Node, *, catch_assertions: bool = False):
    return compile_(node, jittable=True, catch_assertions=catch_assertions)


def nonjit_compile(node: Node, *, catch_assertions: bool = False):
    return compile_(node, jittable=False, catch_assertions=catch_assertions)
