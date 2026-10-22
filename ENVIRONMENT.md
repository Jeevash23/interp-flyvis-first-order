# Environment

Both studies ran in one Python environment: CPython 3.12.14 with flyvis 1.2.0, PyTorch 2.5.1 and torchvision 0.20.1
(CUDA 12.4 builds), NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.11.2, 102 packages in all. The two studies'
`requirements.txt` files pin the same packages at the same versions (only their comment lines differ), so one
environment serves both: `make setup` creates it in `study-1-first-order/.venv` with that study's `make setup`, and
the top-level targets pass it to both studies. Each study's `ENVIRONMENT.md` gives the details, including which
checks need PyTorch and flyvis and which need only NumPy and Matplotlib.

The combined paper's `paper/make_numbers.py` needs only the Python standard library. Building the PDF
(`make paper`) needs a TeX installation with pdfLaTeX, latexmk and the packages loaded in `paper/main.tex`.
