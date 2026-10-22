# Environment

The study ran in the reference study's Python environment: CPython 3.12.14 with flyvis 1.2.0, PyTorch 2.5.1 and
torchvision 0.20.1 (CUDA 12.4 builds), NumPy 1.26.4, SciPy 1.17.1 and Matplotlib 3.11.2, 102 packages in all.
`requirements.txt` pins every one of them, as recorded at the reference study's freeze (setuptools also carries its own
copies of a few small packages, which come with it); every script ran with `python -B`.

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

The figure comparison in the quick check holds with Matplotlib 3.11.2 (other versions write different PDF bytes for
the same figures). The unit tests and the study code import PyTorch and flyvis.

## The FlyVis models

The study simulates the 50 released FlyVis models through the reference study's simulator
(`reference/flyvis-first-order/02-Code/stage1/core.py`), which reads them from the folder named by `FLYVIS_ROOT_DIR`.
They are not included; flyvis's own command downloads them (`flyvis download-pretrained`, with `FLYVIS_ROOT_DIR` set
to `reference/flyvis-first-order/03-Data/flyvis-root`), and `reference/flyvis-first-order/03-Data/stage1/model-selection.json`
holds the SHA-256 of every checkpoint.
