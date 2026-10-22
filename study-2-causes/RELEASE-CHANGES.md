# Changes from the original project

This release was exported from the project folder in which the study ran. Files are copied byte for byte except
those listed below. There are four kinds of change:

- **adapted:** `study/config.json` now names `reference/flyvis-first-order` as the folder of the reference study,
  whose files the code imports (they are shipped there), and the GPU lock script takes its lock file from `GPU_LOCK`.
- **replaced:** the project's `Makefile`, which the freeze lists, is replaced by this release's; the original is kept
  (redacted) as `provenance/Makefile-at-freeze`.
- **redacted:** records in which personal or machine-specific strings were replaced: absolute paths, the external
  drive and other projects' folder names by placeholders (`<project>`, `<reference-project>`, `<portfolio>`,
  `<portfolio-root>`, `<workspace>`, `<research>`, `<external-drive>`, `<drive-archive>`, `<environment>`,
  `<sibling-project>`, `<coordination-folder>`, `<home>`), the operating system's name by `[distro]`, local timestamps
  by the same instant in UTC, and the names of the writing and coding tools in working records by the placeholder `[redacted]` (the assistants used are named in the top-level README.md, under "Use of AI assistants"). Nothing else in these
  files changed; the counts show how many strings of each kind were replaced.
- **not shipped:** files named in a freeze record but left out (listed at the end with their SHA-256).

The nine files under `reference/flyvis-first-order/` are copies of the reference study's files (they are also part of
that study's own release); the study's freeze records them as `ai06_inputs`. Saved outputs are an archive under
`data/saved-outputs/`; `tools/unpack.py` restores it byte for byte and checks each SHA-256 against
`data/PACKED-FILES.json`. Compression is not a change and is not listed below.

To check both freeze records (the study's files and the reference study's inputs) against this release:
`make check-freezes`.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `Makefile` | replaced: this release's Makefile replaces the project's; the original is kept, redacted, as provenance/Makefile-at-freeze | `20e24e24542a41500982af4415b26e4190fe76cc979e45951f67a7ebffdea230` | `study/freeze.json`, `study/freeze-2.json` |
| `provenance/Makefile-at-freeze` | redacted (environment path x1) | `20e24e24542a41500982af4415b26e4190fe76cc979e45951f67a7ebffdea230` | `study/freeze.json`, `study/freeze-2.json` |
| `reference/flyvis-first-order/02-Code/stage1/common.py` | redacted (time zone x1) | `e465cb2e1e16b52433e9a642365f2ca6c910ef8d60dbe1bbe91146de648fff50` | `study/freeze.json`, `study/freeze-2.json` |
| `study/DEVELOPMENT-NOTES.md` | redacted (approval wording x2) | `30b85cc6ff0fea5bcbeb4faf6b36650542a4d1da67b8a226acdc39798b1a5980` | `study/freeze.json`, `study/freeze-2.json` |
| `study/PREDICTIONS-RECORD.json` | redacted (approval wording x1, reference project x1) | `4b4f9bcc513435200c58a1e55e250af8045c6450382c36388c6c78d5ff832135` | `study/freeze.json`, `study/freeze-2.json` |
| `study/PREDICTIONS.md` | redacted (approval wording x1, reference project x1) | `9f3436bae8ac86adae83d257b9c56f515c39a6ebdf473ebd8caa931de0f0c00c` | `study/freeze.json`, `study/freeze-2.json` |
| `study/PROTOCOL-approved-2026-10-16.md` | redacted (approval wording x2, machine details x1, reference project x2) | `b2f5c5b4720a5deaf6ffc238dfcb5253aa640308ad146939e878b400ffd46c93` | `study/freeze.json`, `study/freeze-2.json` |
| `study/PROTOCOL.md` | redacted (AI tool x1, approval wording x3, local clock time x1, machine details x1, reference project x2) | `84822c671692b5569cc64f36f1bb3db64cc8216c84dc2ea556861cd945ed1af2` | `study/freeze.json`, `study/freeze-2.json` |
| `study/amendment-2/apply_amendment2.py` | adapted and redacted: AI tool; local clock time (approval wording x2) | `7781dd8262d8e02813a66214b8e70ed96f15ad3ff229bd6baacbcb05403ed5cb` | `study/freeze-2.json` |
| `study/amendment-2/run_confirmation.sh` | adapted: AI tool and local clock time; absolute path replaced by the release environment; absolute path replaced by the release root | `4c4987ecb22e64e3985e22cef4726d1d40311e07d5d08c26a8e52e7a8cf5daab` | - |
| `study/code/gpu_lock.py` | adapted: the GPU lock file comes from GPU_LOCK (default gpu.lock in the release root) | `b9b970351f54310db4fe4e8a157ef82f7f336862420a2c1f67050328c0bb26ce` | `study/freeze.json`, `study/freeze-2.json` |
| `study/config.json` | adapted: the reference study's files that the code imports are shipped in reference/flyvis-first-order | `b17aa076d5fe74ab57740b189efea44724cead0b8ad51e28927525f2a50629e8` | `study/freeze.json`, `study/freeze-2.json` |
| `study/results/GATE-FAILURE-009.md` | redacted (approval wording x1); shipped inside `data/saved-outputs/study-results.tar.xz` | `b92c42a4972c5f987d0db45c64f6d0dcbe311158c6803e72e64fbacc86b60e26` | `study/freeze-2.json` |
| `study/results/RUNS-LOCATION.md` | redacted (external drive x3, machine details x2, sibling project x1); shipped inside `data/saved-outputs/study-results.tar.xz` | `5097074de6a0eb66f4a4161cb8921f932ff54cd90b8280d9c22d77afdf1cef49` | - |
