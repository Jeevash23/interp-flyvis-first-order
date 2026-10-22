# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `study-results.tar.xz` | 96 | 0.8 | 3.4 | the study's analyses of the development and confirmation models, the gate and completion records, and the JSON sidecars (checksums and metrics) of the four development models' run records (the record arrays not included) |

Total packed size: 0.8 MB. Rendered figures are not included; `make figures` draws them from these outputs.
