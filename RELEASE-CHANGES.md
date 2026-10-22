# How this release was composed

This release combines two code releases, one per study, with the combined manuscript's source:

- `study-1-first-order/` is Study 1's release and `study-2-causes/` is Study 2's release, each copied byte for byte.
  Each was exported from the project folder in which its study ran; its own `RELEASE-CHANGES.md` and
  `release-changes.json` list every file that differs from that project's original, with the original's SHA-256, and
  its own `SHA256SUMS` covers its files.
- `paper/main.tex`, `paper/sections/` and `paper/references.bib` are the combined manuscript's source, copied byte
  for byte.
- `paper/generated/` and `paper/figures/` were written by `paper/make_numbers.py` from the two studies' folders in
  this release. The macro definitions, tables and figures are identical to those the manuscript was built with; only
  the comment lines at the top of `paper/generated/numbers.tex` and the paths in `paper/generated/import-record.json`
  name this release's folders.
- `README.md`, `ENVIRONMENT.md`, `Makefile`, `paper/README.md` and `paper/make_numbers.py` were written for this
  release; `LICENSE` is the same file as in the two studies' releases.

`release-changes.json` lists the two parts with the SHA-256 of their `SHA256SUMS` files, and the nine files that both
parts contain (Study 1's files under `study-2-causes/reference/flyvis-first-order/`), which are identical.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `paper/sections/study2.tex` | redacted (machine details x1) | `bfcb13c338971bb555d98dadc9b8df38db0c680a0a00ab9cd98420f178adda6d` | - |
