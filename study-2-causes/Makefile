# Why first-order estimates misjudge cell-type silencing in the FlyVis models: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` creates. The study's own
# Makefile, part of its freeze, is kept as provenance/Makefile-at-freeze.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
.NOTPARALLEL:

.PHONY: help setup python-check unpack check-freezes quick-check test numbers figures

help:
	@echo "make setup          create .venv with the recorded environment (downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU, under a minute, no model and no network: freeze records, every number and table"
	@echo "                    of the manuscript, the post hoc denominator analysis and the four figures recomputed"
	@echo "make test           the unit tests of the tangents, readouts and scoring (needs PyTorch and flyvis: make setup)"
	@echo "make numbers        rewrite paper/generated/ from the saved outputs"
	@echo "make figures        redraw paper/figures/"

setup:
	uv venv --python 3.12.14 .venv
	grep -v -E '^(torch|torchvision)==' requirements.txt > .venv/requirements-pypi.txt
	uv pip install --python .venv/bin/python -r .venv/requirements-pypi.txt --no-deps
	uv pip install --python .venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/cu124 \
	  'torch==2.5.1+cu124' 'torchvision==0.20.1+cu124'

python-check:
	@$(PY) -c "import numpy, matplotlib" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

check-freezes: unpack
	$(PY) tools/check_freezes.py

quick-check: unpack
	CUDA_VISIBLE_DEVICES= $(PY) tools/check_freezes.py
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_numbers.py --check
	CUDA_VISIBLE_DEVICES= $(PY) tools/replay_analyses.py

test: unpack
	CUDA_VISIBLE_DEVICES= $(PY) -B study/code/test_arms.py

numbers: unpack
	$(PY) -B paper/make_numbers.py

figures: unpack
	$(PY) -B paper/make_figures.py
