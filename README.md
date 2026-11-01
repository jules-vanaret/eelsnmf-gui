# eelsnmf-gui

GUI for [EELS NMF](https://github.com/PauTorru/EELSNMF) processing, including a standalone Qt GUI and
reusable processing utilities for scripts and batch workflows.

## Features

- Qt GUI for end-to-end EELS NMF workflows (preprocess, masking, decomposition,
	plotting, exports).
- Scriptable processing API in `src/eelsnmf_gui/_processing.py` for
	batch usage outside the GUI.
- Caching pipeline that reuses preprocessing outputs across plots and runs to
	avoid redundant work.

## Design Notes

- Preprocessing, masking, and decomposition are separated into explicit stages
	so each stage can be reused independently.
- A shared cache dictionary is passed through the pipeline and keyed by
	parameter subsets to enable partial invalidation.
- The GUI is a thin wrapper around the processing utilities to keep behavior
	consistent between interactive and scripted runs.

## Installation

Install parent package from source:
```
pip install "EELSNMF[EML] @ git+https://github.com/PauTorru/EELSNMF.git"
```

Then install package from source:

```
git clone https://github.com/jules-vanaret/eelsnmf_gui
cd eelsnmf-gui
pip install -e .
```

If you want GPU support, make sure to install [cupy](https://docs.cupy.dev/en/stable/install.html) (we recommend using the conda installer or equivalent to minimize dependency issues):
```
conda install -c conda-forge cupy
```

## Run the GUI:

Using the console entry point:
```
eelsnmf-gui
```
or calling the script directly (make sure to run from the package root):
```
python run_gui.py
```

## License

Distributed under the terms of the BSD-3 license.
