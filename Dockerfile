# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG DREAL_PYTHON_VERSION=4.21.6.2
ARG RUST_VERSION=1.88.0

ENV VIRTUAL_ENV=/opt/zolotone-venv
ENV CARGO_HOME=/opt/cargo
ENV RUSTUP_HOME=/opt/rustup
ENV PATH=/opt/zolotone-venv/bin:/opt/cargo/bin:${PATH}
ENV PYTHONUNBUFFERED=1
ENV GMP_MPFR_SYS_USE_SYSTEM_LIBS=1

# Zolotone uses Python 3.11 and dReal's native IBEX/CLP dependencies. Ubuntu
# 22.04 is used because it is supported by dReal's package repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && add-apt-repository ppa:dreal/dreal -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bison \
        coinor-libclp-dev \
        libfl-dev \
        libgmp-dev \
        libibex-dev \
        libmpfr-dev \
        libnlopt-cxx-dev \
        m4 \
        pkg-config \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip setuptools \
    && python -m pip install --no-cache-dir "wheel<0.38" \
    && python -m pip install --no-cache-dir --no-build-isolation \
        "dreal==${DREAL_PYTHON_VERSION}"

# The pinned Rival3 revision uses let chains, which require Rust 1.88 or newer.
# Pin that toolchain and build the bridge's locked dependency graph.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
          --default-toolchain "${RUST_VERSION}"

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir maturin

COPY crates/rival_bridge/ crates/rival_bridge/
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    --mount=type=cache,target=/app/crates/rival_bridge/target \
    python -m maturin build \
        --release \
        --locked \
        --manifest-path crates/rival_bridge/Cargo.toml \
        --out /tmp/rival-wheel \
    && python -m pip install --no-cache-dir /tmp/rival-wheel/*.whl \
    && rm -rf /tmp/rival-wheel

COPY . .

RUN chmod +x /app/infra/docker-run-designs.sh \
    && mkdir -p /reports \
    && chmod 0777 /reports \
    && python -c "import _rival3, dreal, egglog, z3"

VOLUME ["/reports"]
ENTRYPOINT ["/app/infra/docker-run-designs.sh"]
