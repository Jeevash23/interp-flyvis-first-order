# The preprint's source

*First-order estimates find the influential cell types in the FlyVis models but misjudge their silencing effects: the
nonlinearity is in the readout* (the preprint, version of 19 November 2026). Files that differ from the preprint's source, by the same wording rules as the rest of the release: `sections/study2.tex` (machine details x1).

| Path | What it is |
|---|---|
| `main.tex` | Preamble and the order of the sections |
| `sections/` | `abstract`, `introduction`, `methods`, `study1`, `study2` (with the cost section), `related`, `discussion` (with the conclusion), `appendix` |
| `references.bib` | The union of the two studies' bibliographies, without the two entries in which each cited the other |
| `generated/numbers.tex` | The union of the two studies' generated macros (written by `make_numbers.py`) |
| `generated/s1-*.tex`, `generated/s2-*.tex` | The two studies' generated table bodies |
| `generated/import-record.json` | Where every macro, table and figure came from |
| `figures/` | Seven of the two studies' eight figures (Study 2's strength figure is not used) |
| `make_numbers.py` | Writes `generated/` and `figures/` from `../study-1-first-order/paper/` and `../study-2-causes/paper/`; `--check` compares |

Build with `make paper` from the release folder (or `latexmk -pdf main.tex` here). Every number in the text is a
macro from `generated/numbers.tex` except a few constants (strengths, the 95% level, quoted values of cited papers and
the times in Study 2's record), which appear in the same form in the two studies' own manuscripts.
