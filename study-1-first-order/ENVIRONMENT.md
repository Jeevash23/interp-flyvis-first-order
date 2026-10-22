# Environment

The frozen study ran in one Python environment: CPython 3.12.14 with flyvis 1.2.0, PyTorch 2.5.1 and torchvision 0.20.1
(CUDA 12.4 builds), NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.11.2, 102 packages in all. `requirements.txt` pins
every one of them, as recorded at the freeze in `provenance/revision-environment.json` (the record also lists the
copies of 16 small packages that setuptools carries inside itself; they come with setuptools and are not pinned); the
lock file of the earlier stage is `02-Code/stage1/requirements-cuda.lock.txt`.

## A fresh install

```sh
make setup
```

creates `.venv` with uv's managed CPython 3.12.14, installs the pinned packages from PyPI and then PyTorch and
torchvision from PyTorch's CUDA 12.4 index (`https://download.pytorch.org/whl/cu124`), all without dependency
resolution, so the environment is exactly the recorded one. It includes PyTorch's CUDA libraries (several GB).

## A smaller environment

The quick check needs only NumPy and Matplotlib:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv numpy==1.26.4 scipy==1.17.1 matplotlib==3.11.2
```

`paper/make_figures.py --check` compares the drawn figures with the shipped ones byte for byte, which holds with
Matplotlib 3.11.2 (other versions write different PDF bytes for the same figures). The unit tests and the study
code import PyTorch and flyvis.

## The FlyVis models

The study used the 50 released FlyVis models (Lappalainen et al., 2024) through the flyvis package, which reads them
from the folder named by `FLYVIS_ROOT_DIR`; the study code sets it to `03-Data/flyvis-root/`. They are not included.
`make flyvis-models` downloads the released archive there with flyvis's own command (`flyvis download-pretrained`);
flyvis builds the connectome tables and stimulus renderings in the same folder on first use.
`03-Data/stage1/model-selection.json` and `study/freeze.json` hold the SHA-256 of every checkpoint and of the other
frozen FlyVis files, and the study code refuses a checkpoint whose digest differs. Three files of the installed flyvis
package and the connectome file it ships (`fib25-fib19_v2.2.json`) are among the frozen files; they are part of
flyvis 1.2.0.
