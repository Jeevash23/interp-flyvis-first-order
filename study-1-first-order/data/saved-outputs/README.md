# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `stage1-results.tar.xz` | 1302 | 2.4 | 12.6 | the earlier stage: run records, analyses, benchmarks and verification (response arrays not included, except two intact-network responses the frozen unit tests read) |
| `study-results.tar.xz` | 15184 | 23.3 | 159.2 | the frozen study: analysis tables of the development, confirmation and pooled sets, verification records, and for each of the 50 models and two stimuli every JSON response record with the intact network's responses and the derivatives (the 14,560 silencing response arrays not included) |

Total packed size: 25.6 MB. Rendered figures are not included; `make figures` draws them from these outputs.
