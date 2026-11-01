from qtpy.QtWidgets import QApplication

from ._widget import EELSProcessingWidget


def main() -> None:
    app = QApplication.instance() or QApplication([])
    widget = EELSProcessingWidget(debug=False)
    widget.setWindowTitle("EELS NMF GUI")
    widget.show()
    app.exec_()
