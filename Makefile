PYTHON ?= python3.11
VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3.11
VENV_PIP := $(VENV_PYTHON) -m pip

REPORTS_DIR ?= reports
DESIGNS_REPORT := $(REPORTS_DIR)/run_designs.json
DESIGNS_HTML := $(REPORTS_DIR)/index.html

DREAL_REPO ?= https://github.com/dreal/dreal4

BAZELISK_URL := https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
RUSTUP_INIT_URL := https://sh.rustup.rs
CARGO_HOME ?= $(HOME)/.cargo
RUSTUP_HOME ?= $(HOME)/.rustup
RUST_PATH := $(CARGO_HOME)/bin:$(PATH)
RUST_MIN_VERSION ?= 1.85.0

.PHONY: nightly install install-prereqs unit-tests clean
.PHONY: _check-python _venv _python-deps _install-dreal
.PHONY: _install-rust _install-rival _download_ac_int _bazelisk

_check-python:
	@echo "Checking Python installation"
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "$(PYTHON) not found. Please install Python 3.11."; \
		exit 1; \
	}
	@echo "Python found: $$($(PYTHON) --version)"
	@$(PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info[:2] in {(3, 11)} else 1)' || { \
		echo "$(PYTHON) must be Python  3.11 for dReal compatibility."; \
		exit 1; \
	}

_venv: _check-python
	@echo "Ensuring virtual environment exists in $(VENV_DIR)"
	@if [ ! -x "$(VENV_PYTHON)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	elif [ "$$($(VENV_PYTHON) -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')" != "$$($(PYTHON) -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')" ]; then \
		echo "Recreating $(VENV_DIR) with $(PYTHON)"; \
		$(PYTHON) -m venv --clear $(VENV_DIR); \
	fi


# This runs with sudo
install-prereqs:
	@echo "Installing dReal prerequisites"
	@command -v git >/dev/null 2>&1 || { \
		echo "git not found. Please install git."; \
		exit 1; \
	}
	@set -e; \
	tmp_dir="$$(mktemp -d /tmp/dreal4-prereqs.XXXXXX)"; \
	trap 'rm -rf "$$tmp_dir"' EXIT; \
	echo "Cloning dReal setup repo into $$tmp_dir"; \
	git clone --depth 1 "$(DREAL_REPO)" "$$tmp_dir/dreal4"; \
	cd "$$tmp_dir/dreal4"; \
	uname_s="$$(uname -s)"; \
	if [ "$$uname_s" = "Darwin" ]; then \
		bash ./setup/mac/install_prereqs.sh; \
	elif [ "$$uname_s" = "Linux" ]; then \
		if command -v lsb_release >/dev/null 2>&1; then \
			ubuntu_version="$$(lsb_release -r -s)"; \
		elif [ -r /etc/os-release ]; then \
			. /etc/os-release; \
			ubuntu_version="$$VERSION_ID"; \
		else \
			echo "Cannot detect Ubuntu version"; \
			exit 1; \
		fi; \
		case "$$ubuntu_version" in \
			24.04) \
				sed 's/\<python3-distutils\>//g' ./setup/ubuntu/22.04/install_prereqs.sh > "$$tmp_dir/install_prereqs_24.04.sh"; \
				chmod +x "$$tmp_dir/install_prereqs_24.04.sh"; \
				sudo "$$tmp_dir/install_prereqs_24.04.sh" ;; \
			22.04|20.04) sudo ./setup/ubuntu/$$ubuntu_version/install_prereqs.sh ;; \
			*) echo "Unsupported Ubuntu version: $$ubuntu_version"; exit 1 ;; \
		esac; \
	else \
		echo "Unsupported OS: $$uname_s"; \
		exit 1; \
	fi

clean:
	rm -f "$(DESIGNS_REPORT)" "$(DESIGNS_HTML)"

_python-deps: _venv
	@echo "Installing Python dependencies into $(VENV_DIR)"
	@$(VENV_PIP) install --upgrade pip setuptools
	@$(VENV_PIP) install --upgrade -r requirements.txt

_install-dreal: _venv
	@echo "Installing dReal Python bindings into $(VENV_DIR)"
	@$(VENV_PIP) install --upgrade "wheel<0.38"
	@$(VENV_PIP) install --no-build-isolation dreal

_install-rust:
	@echo "Ensuring Rust/Cargo is available"
	@set -e; \
	install_rustup() { \
		command -v curl >/dev/null 2>&1 || { \
			echo "curl not found. Please install curl or Rust/Cargo manually."; \
			exit 1; \
		}; \
		echo "Installing Rust/Cargo with rustup into $(RUSTUP_HOME) and $(CARGO_HOME)"; \
		echo "This user-local install does not require sudo."; \
		curl --proto '=https' --tlsv1.2 -sSf "$(RUSTUP_INIT_URL)" | CARGO_HOME="$(CARGO_HOME)" RUSTUP_HOME="$(RUSTUP_HOME)" sh -s -- -y --no-modify-path; \
	}; \
	version_at_least() { \
		awk -v current="$$1" -v required="$$2" 'BEGIN { \
			split(current, c, "."); split(required, r, "."); \
			for (i = 1; i <= 3; i++) { \
				c[i] += 0; r[i] += 0; \
				if (c[i] > r[i]) exit 0; \
				if (c[i] < r[i]) exit 1; \
			} \
			exit 0; \
		}'; \
	}; \
	if ! PATH="$(RUST_PATH)" command -v cargo >/dev/null 2>&1 || ! PATH="$(RUST_PATH)" command -v rustc >/dev/null 2>&1; then \
		install_rustup; \
	fi; \
	cargo_version="$$(PATH="$(RUST_PATH)" cargo --version | awk '{print $$2}')"; \
	rust_version="$$(PATH="$(RUST_PATH)" rustc --version | awk '{print $$2}')"; \
	if ! version_at_least "$$cargo_version" "$(RUST_MIN_VERSION)" || ! version_at_least "$$rust_version" "$(RUST_MIN_VERSION)"; then \
		echo "Rust/Cargo must be $(RUST_MIN_VERSION) or newer; found rustc $$rust_version and cargo $$cargo_version"; \
		if PATH="$(RUST_PATH)" command -v rustup >/dev/null 2>&1; then \
			CARGO_HOME="$(CARGO_HOME)" RUSTUP_HOME="$(RUSTUP_HOME)" PATH="$(RUST_PATH)" rustup update stable; \
			CARGO_HOME="$(CARGO_HOME)" RUSTUP_HOME="$(RUSTUP_HOME)" PATH="$(RUST_PATH)" rustup default stable; \
		else \
			install_rustup; \
		fi; \
	fi; \
	echo "Rust found: $$(PATH="$(RUST_PATH)" rustc --version)"; \
	echo "Cargo found: $$(PATH="$(RUST_PATH)" cargo --version)"

_install-rival: _venv _install-rust
	@echo "Installing Rival3 Python bridge into $(VENV_DIR)"
	@$(VENV_PIP) install --upgrade maturin
	@env -u CONDA_PREFIX VIRTUAL_ENV="$(abspath $(VENV_DIR))" PATH="$(RUST_PATH)" $(VENV_PYTHON) -m maturin develop -m crates/rival_bridge/Cargo.toml

_download_ac_int:
	@echo "Downloading ac_int library into infra/ac_types"
	@if [ ! -d "infra/ac_types" ]; then \
		git clone https://github.com/hlslibs/ac_types.git infra/ac_types; \
	fi


# This runs without sudo
install: _python-deps _install-dreal _install-rival _download_ac_int

unit-tests:
	@echo "Running infra/unittests.py..."
	@$(VENV_PYTHON) -m infra.unittests
	@echo "Complete"

# Bazelisk is a non-sudo version of Bazel used for nightly.
_bazelisk:
	mkdir -p "$$HOME/.local/bin"; \
	curl -L $(BAZELISK_URL) -o "$$HOME/.local/bin/bazel"; \
	chmod +x "$$HOME/.local/bin/bazel"

nightly: clean _bazelisk install
	@echo "Running design checks..."; \
	$(VENV_PYTHON) infra/run_designs.py --report "$(DESIGNS_REPORT)"; \
	$(VENV_PYTHON) infra/make_designs_html.py --report-dir "$(REPORTS_DIR)"; \
	echo "Reports written to $(REPORTS_DIR)"; \
