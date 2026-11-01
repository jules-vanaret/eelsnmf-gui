from qtpy.QtWidgets import QApplication

from eelsnmf_gui import EELSProcessingWidget


def main():
    app = QApplication.instance() or QApplication([])
    widget = EELSProcessingWidget(debug=False)
    widget.setWindowTitle("EELS NMF GUI")
    widget.show()
    app.exec_()


if __name__ == "__main__":
    main()
