# First-order estimates of cell-type silencing in the FlyVis models: code release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 1
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
.NOTPARALLEL:

.PHONY: help setup python-check unpack check-freezes quick-check test numbers figures flyvis-models

help:
	@echo "make setup          create .venv with the recorded environment (downloads packages, see ENVIRONMENT.md)"
	@echo "make quick-check    CPU, under a minute, no model and no network: freeze records, every number and table"
	@echo "                    of the manuscript, its four figures and the post hoc summaries recomputed and compared"
	@echo "make test           the frozen unit tests of the scoring code (needs PyTorch: make setup)"
	@echo "make numbers        rewrite paper/generated/ from the saved outputs"
	@echo "make figures        redraw changed figures in paper/figures/"
	@echo "make flyvis-models  download the released FlyVis models into 03-Data/flyvis-root/ (for model reruns)"

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
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_numbers.py --check && echo "paper/make_numbers.py --check: every macro and table matches"
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_figures.py --check
	CUDA_VISIBLE_DEVICES= $(PY) -B analysis/posthoc_summaries.py --check

test: unpack
	CUDA_VISIBLE_DEVICES= $(PY) -B study/code/test_analysis.py

numbers: unpack
	$(PY) -B paper/make_numbers.py

figures: unpack
	$(PY) -B paper/make_figures.py

# The released checkpoints of the 50 models (the FlyVis authors' archive); flyvis builds the connectome tables and
# stimulus renderings in the same folder on first use. The study code sets FLYVIS_ROOT_DIR to this folder.
flyvis-models: python-check
	mkdir -p 03-Data/flyvis-root
	FLYVIS_ROOT_DIR=$(CURDIR)/03-Data/flyvis-root $(dir $(PY))flyvis download-pretrained
