# First-order estimates find the influential cell types in the FlyVis models but misjudge their silencing effects

Code and results for the study *First-order estimates find the influential cell types in the FlyVis models but misjudge their silencing effects: the nonlinearity is in the readout*, by Richard Jeevash. The study is Chapter 8 of the thesis [*What activation patching shows*](https://doi.org/10.5281/zenodo.24555440), which is the version to read and cite.

This repository holds the code, frozen inputs, study records and saved outputs behind it, and the preprint's LaTeX
source; the saved outputs and the frozen protocols are attached to its v1.0 release (see [Release
files](#release-files)). The paper reports two studies of the same 50 released FlyVis models, stimuli and readouts.
Every number, generated table and figure in the paper can be recomputed from the saved outputs on a CPU, without a
model or network (`make quick-check`).

## Release files

The saved outputs and the frozen protocols are not in this repository's files; they are attached to its v1.0 release:

| File | Size | Contents |
|---|---|---|
| `per-run-outputs.tar.gz` | 26.5 MB | The saved outputs: the 3 compressed archives in `study-1-first-order/data/saved-outputs/` and `study-2-causes/data/saved-outputs/`, which `make unpack` places in the study folders and checks file by file |
| `frozen-protocols.tar.gz` | 43 kB | The studies' protocols, predictions, amendments, approvals and freeze records (23 files), at their places in the folder, where the checks read them |
| `SHA256SUMS` | | Checksums of the two archives |

Download the three files into an empty folder, check them there, and unpack them into this one:

```sh
sha256sum -c SHA256SUMS
tar -xzf per-run-outputs.tar.gz -C /path/to/interp-flyvis-first-order
tar -xzf frozen-protocols.tar.gz -C /path/to/interp-flyvis-first-order
```

After unpacking, `sha256sum -c SHA256SUMS` in this folder checks every file that its `SHA256SUMS` lists, the
unpacked attachments included (each part keeps its own list), and the checks below run as described.

## What the paper asks

**Study 1** compares first-order (gradient) estimates of silencing each of the 65 cell types with exact silencing,
scored by the change in direction selectivity of the eight T4 and T5 subtypes, and with rankings from the connectome,
the trained weights and the input currents. The estimates find the influential cell types well but misjudge the size
and sign of their effects. **Study 2** asks why, changing one thing at a time on the same models: mean replacement
instead of silencing to zero, an index built from mean instead of peak responses, and the second-order term of the
Taylor expansion; it also removes one source of nonlinearity at a time from Study 1's estimate (the network's or the
readout's). Both studies fixed their analysis on four development models before analysing the other 46.

## What is in this folder

| Path | Contents |
|---|---|
| `study-1-first-order/` | Study 1's complete release: protocol, frozen code, freeze records, saved outputs, the generators of its numbers, tables and figures, and its own README, `ENVIRONMENT.md`, `RELEASE-CHANGES.md`, `SHA256SUMS` and Makefile |
| `study-2-causes/` | Study 2's complete release, organised the same way. Its `reference/flyvis-first-order/` holds the nine files of Study 1 that its code imports or its freeze names; they are identical to the same files in `study-1-first-order/` |
| `paper/` | The preprint's source: `main.tex`, `sections/`, `references.bib`, `generated/` and `figures/` (copies of the two studies' generated macros, tables and figures), and `make_numbers.py`, which rebuilds and checks those copies |
| `ENVIRONMENT.md` | The environment of both studies and of the paper |
| `RELEASE-CHANGES.md`, `release-changes.json` | How this folder was composed |
| `SHA256SUMS` | The SHA-256 of every file |
| `LICENSE` | The MIT licence (each part carries the same file) |

## Checking

First unpack the two release attachments into this folder (see [Release files](#release-files)): the checks read
the saved outputs and the freeze records they contain.

```sh
make setup        # once: the recorded environment (see ENVIRONMENT.md)
make quick-check  # both studies' quick checks, then the paper's imports
```

`make quick-check` runs `make quick-check` in each study's folder (each unpacks its saved outputs, checks its freeze
records and recomputes every number, table and figure of its own manuscript) and then
`paper/make_numbers.py --check`, which confirms that the combined paper's macros, tables and figures are exactly the
two studies' and that the quantities both studies report agree. Each study's README describes its longer checks and
how to rerun the simulations, which need the released FlyVis models and a GPU.

The freezes are local hash records written by the study code, not public preregistrations.

## Licence

MIT (`LICENSE`), covering the code and the studies' records and saved outputs; each part carries the same file. Model weights, tokenisers and packages are not included and keep their own licences.

## Study codes

The records and code name the studies by the codes they had while the work was done. Each code that occurs here,
and where that study is released:

| Code | Released in |
|---|---|
| AI-01 | `interp-answer-repair` |
| AI-03 | `interp-donor-pooling` |
| AI-05 | `interp-joint-vs-single` |
| AI-06 | this repository |
| AI-09 | this repository |

## Use of AI assistants

As the preprint states:

> Language-model assistants were used throughout this work, under the author's
> direction, to help design the studies, to write and run the code for the experiments and analyses, to draft and
> revise the text and to check the references against their online records. All reported results come from the
> saved study outputs, and the numbers in the text were generated from those outputs or checked against them.

In the working records shipped here, the names of software tools were replaced by placeholders such as
`[redacted]`.

## Citation

Please cite the thesis *What activation patching shows* (Chapter 8, [doi:10.5281/zenodo.24555440](https://doi.org/10.5281/zenodo.24555440)) and, for the code and results, this
repository; [CITATION.cff](CITATION.cff) has the details.

```bibtex
@software{interp_flyvis_first_order_2026,
  author  = {Jeevash, Richard},
  title   = {{First-order estimates find the influential cell types in the FlyVis models but misjudge their silencing effects: code and results}},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/Jeevash23/interp-flyvis-first-order}
}
```
