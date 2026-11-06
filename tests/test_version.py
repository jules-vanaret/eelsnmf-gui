import eelsnmf_gui


def test_version_is_defined():
    """The package exposes a static version string from pyproject.toml."""
    assert eelsnmf_gui.__version__ not in ("", "unknown")
    assert eelsnmf_gui.__version__.count(".") >= 1
