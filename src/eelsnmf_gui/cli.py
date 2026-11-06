import argparse

from qtpy.QtWidgets import QApplication

from ._widget import EELSProcessingWidget


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="eelsnmfgui",
        description="GUI for EELS NMF processing.",
    )
    parser.add_argument(
        "--pre",
        action="store_true",
        help=(
            "show the preprocessing controls and preprocess the loaded "
            "spectrum images (ZLP alignment, despiking). Without this flag, "
            "the loaded SIs are assumed already preprocessed."
        ),
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "allow joint processing of several folders (shows the "
            "'Process folders' selector). Without this flag, folders are "
            "processed individually."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    app = QApplication.instance() or QApplication([])
    widget = EELSProcessingWidget(
        debug=False, pre_mode=args.pre, joint_mode=args.joint
    )
    title = "EELS NMF GUI"
    if args.pre:
        title += " (preprocessing)"
    if args.joint:
        title += " (joint)"
    widget.setWindowTitle(title)
    widget.show()
    app.exec_()
