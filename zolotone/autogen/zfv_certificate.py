"""Compact ZFV certificates for exhaustive checks.

Runs store inclusive packed assignment intervals; JSON displays exact named boxes. The
certificate records completed verification, not a formal proof or cached outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal


FORMAT = "zolotone-zfv-v1"
PRECISION_POLICY = "maximum-observed-precision"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_digest(package: Path):
    files = [
        (f"zolotone/{path.relative_to(package).as_posix()}", path)
        for path in package.rglob("*.py")
        if "__pycache__" not in path.relative_to(package).parts
    ]
    return _sha256(_json([
        (name, hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in sorted(files)
    ]))


def _checker_identity():
    from ..mpfi.interval import native as mpfi_native
    from ..rival.machine import _load_native_module

    package = Path(__file__).resolve().parents[1]
    return {
        "sources_sha256": _source_digest(package),
        "rival_bridge_sha256": hashlib.sha256(Path(_load_native_module().__file__).read_bytes()).hexdigest(),
        "mpfi_bridge_sha256": hashlib.sha256(Path(mpfi_native.__file__).read_bytes()).hexdigest(),
    }


def _encode(value):
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
        return [_encode(item) for item in value]
    if is_dataclass(value):
        return {
            "node": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"Cannot serialize certificate value: {type(value).__name__}")


def _dtype(dtype):
    result = {"kind": type(dtype).__name__, "total_bits": dtype.total_bits()}
    if hasattr(dtype, "frac_bits"):
        result["frac_bits"] = dtype.frac_bits
    return result


def _decode_assignment(index, input_widths):
    """Mixed-radix decoding matching lexicographic Cartesian-product order."""
    raw_inputs = []
    for width in reversed(input_widths):
        raw_inputs.append(index & ((1 << width) - 1))
        index >>= width
    return tuple(reversed(raw_inputs))


def _manifest(spec, component, inputs, expanded_choices, rounding_choices, ulp, error_bound, inclusive):
    """Bind verification to the original spec, choices, input coding, and C++ source."""
    specification = {
        "root": _encode(spec.spec_ast),
        "inputs": [_encode(source) for source, _ in inputs],
        "return_annotation": _encode(spec.return_annotation),
        "assumptions": _encode(spec.spec_ctx.assumes),
        "checks": _encode(spec.spec_ctx.checks),
        "requirements": _encode(spec.spec_ctx.requirements),
        "rounding_choices": [[_encode(node), _encode(choice)] for node, choice in spec.rounding_choices.items()],
    }
    source = component.to_cpp("lowered_entry", jittable=True)
    total_input_bits = sum(dtype.total_bits() for _, dtype in inputs) + rounding_choices
    enumeration_inputs = [{"name": variable.name, **_dtype(dtype)} for variable, dtype in inputs] + [
        {
            "name": expanded_choices[index].name if index < len(expanded_choices) else f"choice_{index}",
            "kind": "Bool", "total_bits": 1,
        }
        for index in range(rounding_choices)
    ]
    # This describes the checker semantics, rather than pinning to an absolute
    # checkout path or requiring an identical Python installation.
    return {
        "spec_sha256": _sha256(_json(specification)),
        "generated_cpp_sha256": _sha256(source),
        "specification": specification,
        "checker": _checker_identity(),
        "enumeration": {
            "inputs": enumeration_inputs,
            "inputs_order": [item["name"] for item in enumeration_inputs],
            "total_input_bits": total_input_bits,
            "total_assignments": 1 << total_input_bits,
        },
        "obligations": {
            "output": _dtype(component.dtype),
            "rounding_mode": spec.spec_ast.mode.value,
            "ulp": str(ulp),
            "error_bound": str(error_bound),
            "inclusive": inclusive,
        },
        "precision_policy": PRECISION_POLICY,
    }


@dataclass(frozen=True)
class ZFVRun:
    start: int
    stop: int
    outcome: Literal["valid", "rejected"]
    replay_precision: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.stop) is not int or not 0 <= self.start <= self.stop:
            raise ValueError("Certificate runs require 0 <= start <= stop")
        if self.outcome not in ("valid", "rejected"):
            raise ValueError("Certificate outcome must be valid or rejected")
        if type(self.replay_precision) is not int or not 20 <= self.replay_precision <= 2**32 - 1:
            raise ValueError("Certificate replay_precision must be an integer between 20 and 2**32 - 1")

    @property
    def num_assignments(self):
        return self.stop - self.start + 1


class _RunCollector:
    def __init__(self):
        self.runs = []
        self.next_index = 0

    def append(self, index, outcome, precision):
        if index != self.next_index:
            raise ValueError("Certificate assignments must be consecutive")
        run = ZFVRun(index, index, outcome, precision)
        if self.runs and self.runs[-1].outcome == outcome:
            previous = self.runs[-1]
            self.runs[-1] = replace(previous, stop=index, replay_precision=max(previous.replay_precision, precision))
        else:
            self.runs.append(run)
        self.next_index += 1


def _validate_coverage(runs, total):
    next_index = 0
    for run in sorted(runs, key=lambda run: run.start):
        run.__post_init__()
        if run.start != next_index or run.stop >= total:
            raise ValueError("Certificate runs must cover all assignments without gaps or overlaps")
        next_index = run.stop + 1
    if next_index != total:
        raise ValueError("Certificate runs must cover all assignments without gaps or overlaps")


def _run_boxes(run, order, widths):
    """Split a packed interval into boxes without visiting individual points."""
    if not order:
        yield {"inputs": {}, "outcome": run.outcome, "replay_precision": run.replay_precision}
        return
    start = run.start
    while start <= run.stop:
        point = _decode_assignment(start, widths)
        suffix_bits = sum(widths)
        for axis, width in enumerate(widths):
            suffix_bits -= width
            stride = 1 << suffix_bits
            # Free lower inputs require alignment to a complete suffix block.
            if start % stride:
                continue
            # Vary this input without wrapping its radix or passing run.stop.
            # The last axis has stride 1, so a valid interval always advances.
            length = min((1 << width) - point[axis], (run.stop - start + 1) // stride)
            if length == 0:
                continue
            bounds = [
                [value, value] if index < axis else
                [value, value + length - 1] if index == axis else
                [0, (1 << widths[index]) - 1]
                for index, value in enumerate(point)
            ]
            yield {
                "inputs": dict(zip(order, bounds, strict=True)),
                "outcome": run.outcome, "replay_precision": run.replay_precision,
            }
            start += length * stride
            break


def _enumeration_layout(enumeration):
    if set(enumeration) != {"inputs", "inputs_order", "total_input_bits", "total_assignments"}:
        raise ValueError("Invalid certificate enumeration fields")
    names = [item["name"] for item in enumeration["inputs"]]
    order = enumeration["inputs_order"]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("Certificate input names must be unique strings")
    if (not isinstance(order, list) or any(not isinstance(name, str) for name in order)
            or len(order) != len(names) or set(order) != set(names)):
        raise ValueError("Certificate inputs_order must contain each input name exactly once")
    total = enumeration["total_assignments"]
    if type(total) is not int or total < 1:
        raise ValueError("Invalid certificate total_assignments")
    widths = [item["total_bits"] for item in enumeration["inputs"]]
    if any(type(width) is not int or not 1 <= width <= 64 for width in widths):
        raise ValueError("Invalid certificate input width")
    if any(item["kind"] == "Bool" and item["total_bits"] != 1 for item in enumeration["inputs"]):
        raise ValueError("Certificate Boolean inputs must have one bit")
    total_input_bits = enumeration["total_input_bits"]
    if type(total_input_bits) is not int or total_input_bits != sum(widths):
        raise ValueError("Certificate total_input_bits disagrees with input widths")
    if total != 1 << total_input_bits:
        raise ValueError("Certificate total disagrees with input widths")
    by_name = dict(zip(names, widths, strict=True))
    return order, tuple(by_name[name] for name in order), total


def _box_run(box, order, widths):
    """Read the contiguous boxes emitted by _run_boxes back into intervals."""
    if set(box) != {"inputs", "outcome", "replay_precision"}:
        raise ValueError("Invalid certificate JSON run fields")
    if not isinstance(box["inputs"], dict) or set(box["inputs"]) != set(order):
        raise ValueError("Certificate runs must name every input exactly once")
    start = stop = 0
    volume = 1
    for name, width in zip(order, widths, strict=True):
        bounds = box["inputs"][name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("Certificate input ranges require [start, stop] arrays")
        low, high = bounds
        if type(low) is not int or type(high) is not int or not 0 <= low <= high < 1 << width:
            raise ValueError(f"Invalid certificate input range for {name!r}")
        start = (start << width) | low
        stop = (stop << width) | high
        volume *= high - low + 1
    # Packing is injective and every box point lies between the packed corners.
    # Equal cardinalities therefore establish that the box has no packed holes.
    if volume != stop - start + 1:
        raise ValueError("Certificate JSON boxes must describe consecutive packed assignments")
    return ZFVRun(start, stop, box["outcome"], box["replay_precision"])


@dataclass(frozen=True)
class ZFVCertificate:
    manifest: dict[str, Any]
    runs: tuple[ZFVRun, ...]

    def __post_init__(self):
        self.validate()

    def validate(self):
        expected_keys = {
            "spec_sha256", "generated_cpp_sha256", "specification", "enumeration",
            "obligations", "precision_policy", "checker",
        }
        if set(self.manifest) != expected_keys:
            raise ValueError("Invalid certificate manifest fields")
        if self.manifest["precision_policy"] != PRECISION_POLICY:
            raise ValueError("Unsupported certificate precision policy")
        for name in ("spec_sha256", "generated_cpp_sha256"):
            digest = self.manifest[name]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Invalid certificate {name}")
        if self.manifest["spec_sha256"] != _sha256(_json(self.manifest["specification"])):
            raise ValueError("Certificate specification hash mismatch")
        obligations = self.manifest["obligations"]
        if set(obligations) != {"output", "rounding_mode", "ulp", "error_bound", "inclusive"}:
            raise ValueError("Invalid certificate obligation fields")
        if type(obligations["inclusive"]) is not bool:
            raise ValueError("Certificate inclusive must be a boolean")
        _, _, total = _enumeration_layout(self.manifest["enumeration"])
        _validate_coverage(self.runs, total)

    @property
    def checked(self):
        return sum(run.num_assignments for run in self.runs if run.outcome == "valid")

    @property
    def skipped(self):
        return sum(run.num_assignments for run in self.runs if run.outcome == "rejected")

    def decode_assignment(self, index: int):
        """Return raw input codes, including Boolean choices, in input order."""
        enumeration = self.manifest["enumeration"]
        if type(index) is not int or not 0 <= index < enumeration["total_assignments"]:
            raise ValueError("Assignment index is outside the certificate domain")
        widths = {item["name"]: item["total_bits"] for item in enumeration["inputs"]}
        order = enumeration["inputs_order"]
        values = _decode_assignment(index, tuple(widths[name] for name in order))
        by_name = dict(zip(order, values, strict=True))
        return tuple(by_name[item["name"]] for item in enumeration["inputs"])

    def to_dict(self):
        # Frozen fields can still contain dictionaries modified since construction.
        self.validate()
        order, widths, _ = _enumeration_layout(self.manifest["enumeration"])
        # Return a detached JSON value, including for nested manifest fields.
        document = json.loads(_json({
            "format": FORMAT, **self.manifest,
            "runs": [box for run in self.runs for box in _run_boxes(run, order, widths)],
            "num_valid_assignments": self.checked, "num_rejected_assignments": self.skipped, "complete": True,
        }))
        for descriptor in [*document["enumeration"]["inputs"], document["obligations"]["output"]]:
            if "frac_bits" in descriptor:
                descriptor["frac_bits"] = descriptor.pop("frac_bits")
        for run in document["runs"]:
            run["inputs"] = {
                name: run["inputs"][name] for name in document["enumeration"]["inputs_order"]
            }
        return document

    @classmethod
    def from_dict(cls, document):
        document = json.loads(_json(document))
        if not isinstance(document, dict):
            raise ValueError("A ZFV certificate must be a JSON object")
        if document.pop("format", None) != FORMAT or document.pop("complete", None) is not True:
            raise ValueError("Unsupported or incomplete ZFV certificate")
        try:
            checked, skipped = document.pop("num_valid_assignments"), document.pop("num_rejected_assignments")
            order, widths, _ = _enumeration_layout(document["enumeration"])
            intervals = sorted((_box_run(box, order, widths) for box in document.pop("runs")),
                               key=lambda run: run.start)
            runs = []
            for run in intervals:
                if runs and runs[-1].stop + 1 == run.start and runs[-1].outcome == run.outcome:
                    previous = runs[-1]
                    runs[-1] = replace(previous, stop=run.stop,
                                       replay_precision=max(previous.replay_precision, run.replay_precision))
                else:
                    runs.append(run)
            result = cls(document, tuple(runs))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Malformed ZFV certificate") from exc
        if type(checked) is not int or type(skipped) is not int or (checked, skipped) != (result.checked, result.skipped):
            raise ValueError("Certificate counts disagree with its runs")
        return result

    def write_json(self, path: str | Path):
        """Publish only a complete certificate; replace the destination atomically."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def read_json(cls, path: str | Path):
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate certificate key: {key}")
                result[key] = value
            return result

        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream, object_pairs_hook=unique_keys))
