# First-order estimates of cell-type silencing in the FlyVis models, and why they fail: code release.
# Two studies, each a complete release of its own (study-1-first-order/, study-2-causes/), and the combined paper's
# source (paper/). `make help` lists the targets. Both studies use the same environment; PYTHON defaults to the one
# that `make setup` creates in study-1-first-order/.venv.
PYTHON ?= study-1-first-order/.venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
.NOTPARALLEL:

.PHONY: help setup quick-check test paper

help:
	@echo "make setup          create study-1-first-order/.venv with the recorded environment (downloads packages,"
	@echo "                    see ENVIRONMENT.md); both studies use it"
	@echo "make quick-check    CPU, no model and no network: each study's quick check (freeze records, every number,"
	@echo "                    table and figure recomputed from the saved outputs), then the combined paper's imports"
	@echo "make test           each study's unit tests (need PyTorch and flyvis: make setup)"
	@echo "make paper          build paper/main.pdf with latexmk (needs a TeX installation)"

setup:
	$(MAKE) -C study-1-first-order setup

quick-check:
	$(MAKE) -C study-1-first-order quick-check PYTHON=$(PY)
	$(MAKE) -C study-2-causes quick-check PYTHON=$(PY)
	CUDA_VISIBLE_DEVICES= $(PY) -B paper/make_numbers.py --check

test:
	$(MAKE) -C study-1-first-order test PYTHON=$(PY)
	$(MAKE) -C study-2-causes test PYTHON=$(PY)

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex && cp build/main.pdf main.pdf
