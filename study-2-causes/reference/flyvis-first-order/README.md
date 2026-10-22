# Files of the reference study

The study builds on a reference study of the same 50 FlyVis models (first-order estimates of cell-type silencing,
released separately). Its code imports that study's batched simulator and helpers and reads its model selection,
stimulus manifest and network structure; these nine files are copied here, in the reference study's own layout, and
`study/config.json` points at this folder (`ai06_root`). The study's freeze records their SHA-256 as `ai06_inputs`.

| File | What it is |
|---|---|
| `02-Code/stage1/core.py`, `02-Code/stage1/common.py` | The reference study's batched FlyVis simulator and its shared settings |
| `study/code/revision_common.py`, `study/code/analyse_revision.py` | Its paths, stimuli, model list and scoring code |
| `study/config.json`, `study/freeze.json` | Its configuration and freeze record |
| `03-Data/stage1/model-selection.json`, `stimulus-manifest.json`, `network-structure.json` | Its model selection (with the SHA-256 of every checkpoint), stimulus manifest and network structure |

Rerunning the study's model stages needs more of the reference study than this: the FlyVis models, its stimuli and
its saved response records, which its gates compare against (see the README of this release).
