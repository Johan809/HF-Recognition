import sys

from PySide6.QtCore import Qt, QSize, QRect
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
                               QHBoxLayout, QVBoxLayout, QStackedWidget, QButtonGroup)

import theme
from theme import VISUAL_DIR
from widgets import Page, Card, Banner, roundedPixmap
from handHandler import HandPage
from faceHandler import FacePage

HAND_PAGE, FACE_PAGE = 1, 2


def featureIcon(file, crop):
    # the original artwork has the icon and a caption, keep only the icon
    return QPixmap(str(VISUAL_DIR / file)).copy(crop).scaled(
        QSize(72, 72), Qt.KeepAspectRatio, Qt.SmoothTransformation)


class FeatureCard(Card):
    def __init__(self, icon, title, text, tags, onOpen, parent=None):
        super().__init__(parent=parent)
        self.body.setContentsMargins(24, 24, 24, 24)
        self.body.setSpacing(12)

        iconLabel = QLabel()
        iconLabel.setPixmap(icon)
        self.body.addWidget(iconLabel)

        titleLabel = QLabel(title)
        titleLabel.setObjectName('featureTitle')
        self.body.addWidget(titleLabel)

        textLabel = QLabel(text)
        textLabel.setObjectName('featureText')
        textLabel.setWordWrap(True)
        self.body.addWidget(textLabel)

        tagsRow = QHBoxLayout()
        tagsRow.setSpacing(6)
        for tag in tags:
            tagLabel = QLabel(tag)
            tagLabel.setObjectName('tag')
            tagsRow.addWidget(tagLabel)
        tagsRow.addStretch()
        self.body.addLayout(tagsRow)

        self.body.addSpacing(8)
        button = QPushButton('Abrir  →')
        button.setObjectName('primary')
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(onOpen)
        self.body.addWidget(button, 0, Qt.AlignLeft)


class HomePage(Page):
    def __init__(self, goTo, parent=None):
        super().__init__('Bienvenido', 'Elige un módulo para comenzar.', parent)

        # top strip of the original background, it has the project title (without the gear icon)
        hero = QPixmap(str(VISUAL_DIR / 'fondo.jpeg')).copy(QRect(60, 0, 680, 227))
        self.body.addWidget(Banner(hero, 240))

        cards = QHBoxLayout()
        cards.setSpacing(20)
        cards.addWidget(FeatureCard(
            featureIcon('HG.png', QRect(19, 0, 113, 106)),
            'Gestos de la mano',
            'Cuenta los dedos levantados y reconoce 8 gestos en tiempo real usando '
            'segmentación por color de piel y defectos de convexidad.',
            ['OpenCV', 'HSV', 'Convex hull'],
            lambda: goTo(HAND_PAGE)))
        cards.addWidget(FeatureCard(
            featureIcon('FR.png', QRect(44, 0, 110, 106)),
            'Reconocimiento facial',
            'Detecta rostros en fotos o con la webcam y los compara con las personas '
            'registradas en la carpeta img/known.',
            ['face_recognition', 'dlib', 'Webcam'],
            lambda: goTo(FACE_PAGE)))
        self.body.addLayout(cards)
        self.body.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HF Recognition')
        self.setWindowIcon(QIcon(str(VISUAL_DIR / 'logo.png')))
        self.resize(1240, 780)
        self.setMinimumSize(1040, 680)

        root = QWidget()
        rootLayout = QHBoxLayout(root)
        rootLayout.setContentsMargins(0, 0, 0, 0)
        rootLayout.setSpacing(0)
        self.setCentralWidget(root)

        rootLayout.addWidget(self.buildSidebar())

        self.stack = QStackedWidget()
        self.pages = [HomePage(self.goTo), HandPage(), FacePage()]
        for page in self.pages:
            self.stack.addWidget(page)
        rootLayout.addWidget(self.stack, 1)

        self.goTo(0)

    def buildSidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 24, 18, 20)
        layout.setSpacing(6)

        brand = QHBoxLayout()
        brand.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(roundedPixmap(QPixmap(str(VISUAL_DIR / 'logo.png')).copy(QRect(24, 25, 106, 103)),
                                     QSize(42, 42), 10))
        brand.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(0)
        name = QLabel('HF Recognition')
        name.setObjectName('brandName')
        sub = QLabel('Hand & Face')
        sub.setObjectName('brandSub')
        names.addWidget(name)
        names.addWidget(sub)
        brand.addLayout(names)
        layout.addLayout(brand)
        layout.addSpacing(28)

        self.navGroup = QButtonGroup(self)
        self.navButtons = []
        for i, text in enumerate(['Inicio', 'Gestos de la mano', 'Reconocimiento facial']):
            button = QPushButton(text)
            button.setProperty('nav', True)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, index=i: self.goTo(index))
            self.navGroup.addButton(button)
            self.navButtons.append(button)
            layout.addWidget(button)

        layout.addStretch()
        footnote = QLabel('Proyecto final · ITLA\nProgramación III')
        footnote.setObjectName('footnote')
        layout.addWidget(footnote)
        return sidebar

    def goTo(self, index):
        # release the camera of the page we are leaving
        current = self.stack.currentWidget()
        if current is not None and current is not self.pages[index]:
            current.stop()
        self.stack.setCurrentIndex(index)
        self.navButtons[index].setChecked(True)

    def closeEvent(self, event):
        for page in self.pages:
            page.shutdown()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    theme.apply(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
