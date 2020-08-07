import sys
import os
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication

from testGUI import *

class testGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.setWindowIcon(QtGui.QIcon(path of the icon))
        self.ui.btnHand.setStyleSheet("border-image: url(./gui/visual/HG.png);")
        self.ui.btnFace.setStyleSheet("border-image: url(./gui/visual/FR.png);")
        self.ui.btnHand.clicked.connect(self.startHand)
        self.ui.btnFace.clicked.connect(self.startFace)

    def startHand(self):
        os.system('python ./gui/handHandler.py')
        
    def startFace(self):
        os.system('python ./gui/faceHandler.py')
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = testGui()
    GUI.setStyleSheet("border-image: url(./gui/visual/fondo.jpeg);")
    GUI.show()
    sys.exit(app.exec_())