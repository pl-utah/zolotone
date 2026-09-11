import shutil
import tempfile
import unittest
import ctypes
import subprocess
from pathlib import Path

from zolotone import Node, DataType, Tuple

_INFRA_DIR = Path(__file__).resolve().parent
_AC_TYPES_INCLUDE_DIR = _INFRA_DIR / "ac_types" / "include"


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


def _arg_mask(type_: DataType) -> str:
    return str((1 << type_.total_bits()) - 1)


def _wrapper_arg_expr(arg, idx: int, *, jittable: bool) -> str:
    expr = f"arg_{idx}"
    if jittable:
        expr = f"static_cast<{_cpp_abi_type(arg.dtype)}>({expr} & {_arg_mask(arg.dtype)})"
    lowered_type = _lowered_cpp_type(arg.dtype, jittable=jittable)
    return f"static_cast<{lowered_type}>({expr})"


def compile_(node: Node, jittable: bool):
    if not hasattr(node, "inner_args"):
        raise TypeError("compile helpers expect a Primitive or Composite node")

    function_name = "lowered_entry"
    source = node.to_cpp(function_name, jittable=jittable)
    compiler = _find_cpp_compiler()

    tempdir = tempfile.TemporaryDirectory()
    temp_path = Path(tempdir.name)
    header_path = temp_path / "lowered.hpp"
    wrapper_path = temp_path / "wrapper.cpp"
    library_path = temp_path / "lowered.so"

    header_path.write_text(source, encoding="utf-8")
    arg_decls = [
        f"{_cpp_abi_type(arg.dtype)} arg_{idx}"
        for idx, arg in enumerate(node.inner_args)
    ]
    call_args = ", ".join(
        _wrapper_arg_expr(arg, idx, jittable=jittable)
        for idx, arg in enumerate(node.inner_args)
    )
    return_type = _cpp_abi_type(node.dtype)

    wrapper_path.write_text(
        "\n".join(
            [
                "#include <cstdint>",
                '#include "lowered.hpp"',
                "",
                f'extern "C" {return_type} {function_name}_entry({", ".join(arg_decls)}) {{',
                f"    return static_cast<{return_type}>({function_name}({call_args}));",
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
    return tempdir, func


def jit_compile(node: Node):
    return compile_(node, jittable=True)


def nonjit_compile(node: Node):
    return compile_(node, jittable=False)
