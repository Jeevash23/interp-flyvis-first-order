# Data and provenance

`flyvis-root/` contains downloaded public pretrained models, clustering results and generated synthetic-stimulus caches. `FLYVIS_ROOT_DIR` is project-local. Official archives have verified upstream SHA256 checksums in `06-Study-History/*-download.json`. The released ensemble is not trained or altered here. Source code is MIT; a separate weight-data license has not been verified, so no redistribution/public release is performed.

`biology-key.csv` is a 12-row evidence extraction with nine biological/structural claims and three explicitly model-only comparators. It is not a scored model-biology result. Source passages are paraphrased and tied to original figures/conditions.

No Sintel videos, biological recordings, other projects' evaluation data, or user data were used. Synthetic moving-edge inputs use the native tutorial parameters. Failed cache directories are quarantined under Study History, never used as completed data.
