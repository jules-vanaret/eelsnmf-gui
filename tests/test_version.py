import eelsnmf_gui


def test_version_is_defined():
    """The package exposes a static version string from pyproject.toml."""
    assert eelsnmf_gui.__version__ not in ("", "unknown")
    assert eelsnmf_gui.__version__.count(".") >= 1


def test_public_helpers_are_exported():
    """The loader and CuPy helper are part of the public API."""
    from eelsnmf_gui._nmf import cupy_gpu_available
    from eelsnmf_gui._preprocess import load_eml_folders

    assert eelsnmf_gui.cupy_gpu_available is cupy_gpu_available
    assert eelsnmf_gui.load_eml_folders is load_eml_folders
