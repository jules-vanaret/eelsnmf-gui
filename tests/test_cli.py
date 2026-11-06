import pytest

pytest.importorskip("qtpy")

from eelsnmf_gui.cli import _parse_args


def test_parse_args_default_has_no_preprocessing():
    """Without flags the GUI starts in the non-preprocessing mode."""
    args = _parse_args([])
    assert args.pre is False
    assert args.joint is False


def test_parse_args_pre_flag():
    """--pre selects the preprocessing mode."""
    assert _parse_args(["--pre"]).pre is True


def test_parse_args_joint_flag():
    """--joint enables the joint processing UI."""
    assert _parse_args(["--joint"]).joint is True
    assert _parse_args(["--pre", "--joint"]).pre is True
