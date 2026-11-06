# eelsnmf-gui

GUI for [EELS NMF](https://github.com/PauTorru/EELSNMF) processing, including a standalone Qt GUI and
a scriptable processing API for scripts and batch workflows.

## Features

- Qt GUI for end-to-end EELS NMF workflows (preprocess, masking, decomposition,
	plotting, exports).
- `Session` API (`src/eelsnmf_gui/session.py`) exposing the same pipeline
	stages for scripted usage outside the GUI.
- Stage-wise caching: preprocessing, masks, and fitted decompositions are
	reused and invalidated automatically when their inputs change.

## Design Notes

- The pipeline is split into explicit stages: preprocessing (folder loading,
	ZLP alignment, despiking, common energy axis), mask building, model
	building and decomposition, and result analysis (edges, maps, summaries).
- `eelsnmf_gui.session.Session` owns the folders and parameters, caches each
	stage, and self-invalidates:
	- changing folders or the median filter size invalidates everything;
	- changing mask parameters invalidates masks and decompositions, but
		keeps preprocessing;
	- changing decomposition parameters (beam, edge ranges, number of
		components, max iterations, W init) invalidates decompositions only;
	- colors never affect the decomposition: recoloring only re-renders.
- The EELSNMF backend is isolated behind a loader and an engine interface, so
	all other stages run on plain numpy arrays and are testable without
	EELSNMF installed.
- The GUI is a thin wrapper around the `Session`, keeping behavior consistent
	between interactive and scripted runs.

## Installation

Install the EELSNMF backend from source:
```
pip install "EELSNMF[EML] @ git+https://github.com/PauTorru/EELSNMF.git"
```

Then install package from source:

```
git clone https://github.com/jules-vanaret/eelsnmf-gui
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
eelsnmfgui
```
By default the GUI assumes the loaded spectrum images are already
preprocessed (despiked, ZLP-aligned), hides the preprocessing controls, and
processes folders individually. To show the preprocessing controls (and
preprocess the data on load) and/or allow joint processing of several
folders, use:
```
eelsnmfgui --pre --joint
```
or calling the script directly (make sure to run from the package root):
```
python run_gui.py
```

## Testing

Run the test suite with:
```
python -m pytest
```
Unit, session, and widget tests run without EELSNMF installed; the
end-to-end tests marked `integration` run when EELSNMF is available and can
be skipped with `pytest -m "not integration"`.

## License

Distributed under the terms of the BSD-3 license.
