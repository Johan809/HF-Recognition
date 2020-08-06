import sys
import os
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication

class testGui(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('./gui/test.ui', self)
        self.btnHand.clicked.connect(self.startHand)
        self.btnFace.clicked.connect(self.startFace)

    def startHand(self):
        os.system('python ./gui/handHandler.py')
        
    def startFace(self):
        os.system('python ./gui/faceHandler.py')
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = testGui()
    GUI.show()
    sys.exit(app.exec_())

