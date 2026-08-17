# Zolotone

Zolotone is a specification language for floating-point computation. It lets a
model describe arithmetic at the level of mathematical intent while making
floating-point behavior—such as encoding, representable ranges, signed zero,
infinities, and NaNs—explicit.

Models written in the specification language are the **golden models**. A
lower-level implementation can use fixed-point arithmetic, bit manipulation,
decomposed floating-point fields, or optimized datapaths; Zolotone checks that
the implementation refines its golden model.

## Specification model

The specification language builds an immutable symbolic AST with:

- real and Boolean expressions for mathematical relationships;
- explicit assumptions and properties to check;
- floating-point values with observable classifications and format behavior;
- translation to simplification, e-graph, SMT, and nonlinear-real backends.

For example, the golden model for the FP32 IEEE adder states the finite
computation directly while making the special-value behavior explicit:

```python
from zolotone import Cases, case, fp32


def spec_fp32_add(x: "FP32", y: "FP32", ctx):
    nan_case = (
        x.is_nan
        | y.is_nan
        | (x.is_pinf & y.is_ninf)
        | (x.is_ninf & y.is_pinf)
    )
    neg_inf_case = (x.is_ninf | y.is_ninf) & (~nan_case)
    pos_inf_case = (x.is_pinf | y.is_pinf) & (~nan_case)
    neg_zero_case = x.is_nzero & y.is_nzero

    return Cases(
        case(nan_case, fp32.nan()),
        case(neg_inf_case, fp32.ninf()),
        case(pos_inf_case, fp32.inf()),
        case(neg_zero_case, fp32.nzero()),
        case(
            x.is_finite & y.is_finite,
            fp32.encode(value=x.value + y.value, ctx=ctx),
        ),
        ctx=ctx,
    )
```

`Cases` selects the first matching `case` in source order. It requires the
specification context and rejects the specification unless the case conditions
cover every valid input. Floating-point (`FPExpr`) results must be branched with
`Cases`; `If` is limited to scalar `BoolExpr` and `RealExpr` branches.

This model says what the result means. It does not prescribe exponent
alignment, significand formatting, rounding logic, or other implementation
choices. Those belong in `fp32_add`, the implementation model, and are
verified against this golden specification.

## Verification workflow

Zolotone connects golden specifications to typed implementation models:

1. Define math-level intent using `RealExpr`, `BoolExpr`, `FPExpr`, and formats
   such as `fp16`, `fp32`, `bf16`, `e4m3fn`, `e5m2`, `e5m2fnuz`, and `e2m1`.
2. Build an implementation from typed `Primitive` and `Composite` nodes.
3. Attach a specification to each operation or composite.
4. Call `check_determinism()` to prove that repeated evaluations of the
   specification produce equivalent results for the same symbolic inputs.
5. Call `check_spec()` to compare the implementation with its golden model.

Proof obligations can be simplified, rewritten with egglog, discharged with
Z3, and checked with dReal. Floating-point results are split into observable
classification cases so finite values, zeros, infinities, and NaNs are compared
with the appropriate semantics. Both checks accept an optional solver schedule
and return `{"proved": bool, "proof_traces": [...]}`.

Classification cases run concurrently by default. Pass `max_workers=1` to
`check_equivalence()`, `check_spec()`, or `check_determinism()` for serial
verification, or pass a positive integer to bound the spawn-based process
pool. The default uses the process-affinity CPU count when available, then
falls back to `os.cpu_count()` and finally one worker. Set
`ZOLOTONE_MAX_WORKERS` to give command-line verification a machine-wide cap.
Cases are still returned in classification-generation order, while streaming
observer notifications arrive in completion order. Each case worker runs its
solver schedule directly and sequentially; solver-specific budgets such as the
Z3 timeout remain unchanged.

`infra/run_designs.py` also accepts `--max-workers` and a per-check `--timeout`.
Designs and their determinism/specification checks run sequentially, and each
whole check receives one timeout. The `make nightly` defaults are deliberately
bounded to eight workers and 600 seconds per check; set
`DESIGN_MAX_WORKERS` and `DESIGN_TIMEOUT_S` to override them.

## Repository layout

- `zolotone/spec/` — the math-level specification AST, `SpecContext`, and
  floating-point specifications.
- `zolotone/ast/` — typed implementation nodes, composites, and specification
  checking.
- `zolotone/components/` and `zolotone/types/` — fixed-point, floating-point,
  Boolean, tuple, and bit-level building blocks.
- `zolotone/solver/`, `zolotone/smt/`, and `zolotone/egglog/` — proof scheduling
  and solver integrations.
- `zolotone/codegen/` — C++ generation for implementation models.
- `examples/` — FP32 arithmetic, FP32 format converters, conventional/optimized
  BF16 dot products, and reduced WGMMA dot-accumulate models with golden
  specifications. See `examples/converters/README.md` for converter semantics.
- `docs/operators.md` — available implementation operators and primitives.

## Quick start

The development environment uses Python 3.11 because of the dReal bindings.

```sh
make install
make unit-tests
```

`make install` creates `.venv`, installs the Python dependencies and dReal,
builds the Rival3 bridge, and downloads the `ac_int` headers used for generated
C++ tests. It may install a user-local Rust toolchain through rustup when a
suitable Cargo installation is not available.

To inspect and verify the example designs directly:

```sh
.venv/bin/python -m examples.conventional
.venv/bin/python -m examples.optimized
```

Each command builds the typed implementation model, prints its structure,
checks it against the golden specification, and emits a C++ header.

## Docker design reports

The repository includes an image that runs `infra/run_designs.py`, converts
its JSON report to HTML, and writes both files to a host directory. Build it
from the repository root:

```sh
docker build --platform linux/amd64 --tag zolotone .
```

The pinned dReal Python package is available for Linux x86-64. Both commands
explicitly select that platform, so Docker uses x86-64 emulation on ARM hosts.

To build the image and run it with the complete configuration below in one
command, use the Makefile target:

```sh
make run-docker
```

Create the output directory and bind-mount it at `/reports`. Running with the
host user's UID and GID keeps the generated files owned by that user:

```sh
docker run --rm \
  --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$(pwd)/reports",target=/reports \
  --env DESIGN_TIMEOUT_S=1800 \
  zolotone
```

The image includes `tini` as PID 1, so `docker run --init` is unnecessary.
Signals are forwarded to the Python report coordinator and orphaned check
descendants are reaped inside the container.

The image defaults to a 600-second timeout for each check; the command above
and `make run-docker` override it to 1800 seconds. When `DESIGN_MAX_WORKERS` is
omitted, verification uses all CPUs available to the container. Set
`DESIGN_MAX_WORKERS` only when you want to cap parallelism. The Make target's
timeout can be changed with, for example,
`make run-docker DOCKER_TIMEOUT_S=900`.

The report is updated after every completed design, so the mounted JSON file
also retains partial progress if a long run is interrupted. The coordinator
generates partial HTML whenever that JSON report exists before exiting with
status 130. Reusing the same host directory replaces `run_designs.json` and
`index.html` on the next run.

## Reduced WGMMA examples

`examples.wgmma` provides deterministic scalar models of
`fp32.e4m3.e4m3`, `fp32.e5m2.e4m3`, and `fp16.e4m3.e5m2`. Each computes

```text
D = C + a0*b0 + a1*b1 + a2*b2 + a3*b3
```

as an ideal fused operation: FP8 products and accumulation are exact, followed
by one round-to-nearest-even encoding to FP32 or FP16. PTX does not prescribe
the hardware WGMMA accumulation order, rounding, or subnormal behavior, so
these reduced models intentionally define a deterministic contract rather than
the literal K=32 hardware instruction shape. They do not model transpose,
operand scaling, or an optional accumulator enable.

Run any model module directly—for example,
`python -m examples.wgmma_fp32_e4m3_e4m3`—to check it and emit its JIT and
non-JIT headers under `examples/c_models/`. Generated headers are ignored by
Git.

## Rival3 bridge

`zolotone.rival` translates specification expressions into Rival3 for
interval-based feasibility checks. `make install` builds the PyO3 extension;
to rebuild it manually in an activated development environment:

```sh
python -m pip install maturin
python -m maturin develop -m crates/rival_bridge/Cargo.toml
```
