# import system module
import sys
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

# import some Qt modules
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
                               QFileDialog, QMessageBox, QListWidget, QListWidgetItem)
from PySide6.QtGui import QPixmap, QIcon, QFont, QDesktopServices
from PySide6.QtCore import QTimer, Qt, QSize, QUrl

# import Opencv and face recognition modules
import face_recognition as faces
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import theme
from theme import KNOWN_DIR, SCANNED_DIR
from widgets import Page, Card, VideoView, Worker, setVariant, setProp, rgbToImage, roundedPixmap

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# max distance between two encodings to consider them the same person (library default)
TOLERANCE = 0.6

# big photos are reduced before searching faces, it is much faster and precise enough
MAX_IMAGE_SIDE = 1600

KNOWN_COLOR = (31, 181, 201)
UNKNOWN_COLOR = (239, 111, 108)
LABEL_TEXT_COLOR = (11, 17, 22)

KnownFace = namedtuple('KnownFace', 'name encoding avatar')


def loadImage(path, maxSide=None):
    # PIL respects the EXIF orientation of phone photos, face_recognition does not
    img = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if maxSide and max(img.size) > maxSide:
        img.thumbnail((maxSide, maxSide))
    return np.array(img)


def loadKnownFaces():
    """Reads img/known: the file name (without extension) is the name of the person.

    Returns (known faces, names of the files where no face was found).
    """
    known = []
    skipped = []
    for entry in sorted(KNOWN_DIR.iterdir()):
        if entry.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        img = loadImage(entry, MAX_IMAGE_SIDE)
        locations = faces.face_locations(img)
        if not locations:
            skipped.append(entry.name)
            continue
        encoding = faces.face_encodings(img, locations[:1])[0]

        # crop the face with some margin to use it as avatar
        top, right, bottom, left = locations[0]
        pad = (bottom - top) // 3
        avatar = img[max(top - pad, 0):bottom + pad, max(left - pad, 0):right + pad].copy()

        known.append(KnownFace(entry.stem, encoding, avatar))
        print(entry.stem + ' processed')
    return known, skipped


def identify(known, encoding):
    # use the known face with the smallest distance to the new face
    if not known:
        return None
    distances = faces.face_distance([k.encoding for k in known], encoding)
    best = int(np.argmin(distances))
    return known[best].name if distances[best] <= TOLERANCE else None


@lru_cache(maxsize=32)
def labelFont(size):
    return ImageFont.load_default(size=size)


def drawFaces(rgb, detections):
    """Draws a box and a name label for each ((top, right, bottom, left), name) detection."""
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    # draw boxes
    for (top, right, bottom, left), name in detections:
        width = right - left
        draw.rounded_rectangle((left, top, right, bottom), radius=max(4, width // 12),
                               outline=KNOWN_COLOR if name else UNKNOWN_COLOR, width=max(2, width // 60))

    # draw labels after all the boxes, so a box never covers the label of a near face
    for (top, right, bottom, left), name in detections:
        color = KNOWN_COLOR if name else UNKNOWN_COLOR
        width = right - left
        stroke = max(2, width // 60)

        # label below the box (or inside it if there is no room)
        text = name or 'Desconocido'
        font = labelFont(max(12, width // 8))
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        pad = max(4, (b - t) // 3)
        labelHeight = b - t + 2 * pad
        y = bottom + stroke
        if y + labelHeight > pil.height:
            y = bottom - labelHeight
        draw.rounded_rectangle((left, y, left + r - l + 2 * pad, y + labelHeight), radius=pad, fill=color)
        draw.text((left + pad - l, y + pad - t), text, font=font, fill=LABEL_TEXT_COLOR)
    return np.array(pil)


def analyzeImage(path, known):
    img = loadImage(path, MAX_IMAGE_SIDE)

    # find faces in images
    locations = faces.face_locations(img)
    encodings = faces.face_encodings(img, locations)
    detections = [(loc, identify(known, enc)) for loc, enc in zip(locations, encodings)]

    result = drawFaces(img, detections)

    # save the scanned image
    SCANNED_DIR.mkdir(exist_ok=True)
    savedPath = SCANNED_DIR / (Path(path).stem + '_scanned.png')
    Image.fromarray(result).save(savedPath)

    return result, [name for _, name in detections], savedPath


class ReadOnlyList(QListWidget):
    """The user can not select items, the selection only marks the detected people."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.MultiSelection)
        self.setFocusPolicy(Qt.NoFocus)

    def mousePressEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass


class FacePage(Page):
    def __init__(self, parent=None):
        super().__init__('Reconocimiento facial',
                         'Identifica rostros conocidos en una foto o en vivo con la webcam.', parent)
        self.cap = None
        self.known = []
        self.worker = None
        self.processThisFrame = True
        self.lastDetections = []

        content = QHBoxLayout()
        content.setSpacing(20)
        self.body.addLayout(content, 1)

        self.video = VideoView('Analiza una imagen o inicia la webcam\n\n'
                               'Los rostros conocidos se marcan en turquesa y los desconocidos en coral')
        self.video.setMinimumSize(QSize(360, 270))
        content.addWidget(self.video, 1)

        side = QVBoxLayout()
        side.setSpacing(16)
        content.addLayout(side)

        # detected faces
        resultCard = Card('Rostros detectados')
        resultCard.setFixedWidth(300)
        self.count = QLabel('—')
        self.count.setObjectName('bigValue')
        self.count.setAlignment(Qt.AlignCenter)
        self.detail = QLabel('Sin análisis todavía')
        self.detail.setObjectName('muted')
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setWordWrap(True)
        resultCard.body.addWidget(self.count)
        resultCard.body.addWidget(self.detail)
        side.addWidget(resultCard)

        # known people, the detected ones are highlighted
        self.knownCard = Card('Personas conocidas')
        self.knownCard.setFixedWidth(300)
        self.knownList = ReadOnlyList()
        self.knownList.setIconSize(QSize(36, 36))
        self.knownCard.body.addWidget(self.knownList, 1)

        links = QHBoxLayout()
        openFolder = QPushButton('Abrir carpeta')
        openFolder.setObjectName('link')
        openFolder.setCursor(Qt.PointingHandCursor)
        openFolder.setToolTip(str(KNOWN_DIR))
        openFolder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(KNOWN_DIR))))
        self.reloadButton = QPushButton('Recargar')
        self.reloadButton.setObjectName('link')
        self.reloadButton.setCursor(Qt.PointingHandCursor)
        self.reloadButton.clicked.connect(self.loadKnown)
        links.addWidget(openFolder)
        links.addStretch()
        links.addWidget(self.reloadButton)
        self.knownCard.body.addLayout(links)
        side.addWidget(self.knownCard, 1)

        self.imageButton = QPushButton('Analizar imagen…')
        self.imageButton.setObjectName('secondary')
        self.imageButton.setCursor(Qt.PointingHandCursor)
        self.imageButton.clicked.connect(self.uploadImage)
        side.addWidget(self.imageButton)

        self.webcamButton = QPushButton('Iniciar webcam')
        self.webcamButton.setObjectName('primary')
        self.webcamButton.setCursor(Qt.PointingHandCursor)
        self.webcamButton.clicked.connect(self.controlTimer)
        side.addWidget(self.webcamButton)

        # create a timer, each tick processes a single frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.webCamRec)

        self.loadKnown()

    # ---------- known faces ----------

    def setBusy(self, busy):
        for button in (self.imageButton, self.webcamButton, self.reloadButton):
            button.setEnabled(not busy)

    def runInBackground(self, fn, args, onDone):
        self.setBusy(True)
        self.worker = Worker(fn, *args, parent=self)
        self.worker.done.connect(onDone)
        self.worker.failed.connect(self.onWorkerFailed)
        self.worker.finished.connect(lambda: self.setBusy(False))
        self.worker.start()

    def onWorkerFailed(self, message):
        self.setStatus('Ocurrió un error', 'warn')
        QMessageBox.warning(self, 'Error', message)

    def loadKnown(self):
        self.stop()
        self.setStatus('Cargando rostros conocidos…')
        self.runInBackground(loadKnownFaces, (), self.onKnownLoaded)

    def onKnownLoaded(self, result):
        self.known, skipped = result
        self.knownList.clear()
        for person in self.known:
            avatar = roundedPixmap(QPixmap.fromImage(rgbToImage(person.avatar)), QSize(36, 36), 18)
            QListWidgetItem(QIcon(avatar), person.name, self.knownList)
        self.knownCard.findChild(QLabel, 'cardTitle').setText(f'PERSONAS CONOCIDAS · {len(self.known)}')

        if skipped:
            self.setStatus(f'Sin rostro en: {", ".join(skipped)}', 'warn')
        elif not self.known:
            self.setStatus('Agrega fotos a img/known para reconocer personas', 'warn')
        else:
            self.setStatus(f'{len(self.known)} rostros cargados', 'ok')

    def highlightKnown(self, names):
        for i in range(self.knownList.count()):
            item = self.knownList.item(i)
            found = item.text() in names
            font = QFont(item.font())
            font.setBold(found)
            item.setFont(font)
            item.setSelected(found)

    def showDetections(self, names):
        self.count.setText(str(len(names)))
        if not names:
            self.detail.setText('Ningún rostro en la imagen')
        else:
            unknown = names.count(None)
            known = sorted({n for n in names if n})
            parts = known + ([f'{unknown} desconocido' + ('s' if unknown > 1 else '')] if unknown else [])
            self.detail.setText(', '.join(parts))
        self.highlightKnown(set(names))

    # ---------- image ----------

    def uploadImage(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Selecciona una imagen', str(theme.ROOT / 'img'),
                                              'Imágenes (*.jpg *.jpeg *.png *.bmp *.webp)')
        if not path:
            return
        self.stop()
        self.setStatus('Analizando imagen…')
        self.runInBackground(analyzeImage, (path, self.known), self.onImageAnalyzed)

    def onImageAnalyzed(self, result):
        img, names, savedPath = result
        self.video.setFrame(img)
        self.showDetections(names)
        self.setStatus(f'Guardada en scannedImages/{savedPath.name}', 'ok')

    # ---------- webcam ----------

    def webCamRec(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # only process every other frame of video to save time
        if self.processThisFrame:
            # resize frame of video to 1/4 size for faster face recognition processing
            small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
            locations = faces.face_locations(small)
            encodings = faces.face_encodings(small, locations)

            # scale back up face locations since the frame we detected in was scaled to 1/4 size
            self.lastDetections = [(tuple(v * 4 for v in loc), identify(self.known, enc))
                                   for loc, enc in zip(locations, encodings)]
            self.showDetections([name for _, name in self.lastDetections])
        self.processThisFrame = not self.processThisFrame

        self.video.setFrame(drawFaces(rgb, self.lastDetections))

    # start/stop timer
    def controlTimer(self):
        if self.timer.isActive():
            self.stop()
            return

        # create video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            QMessageBox.warning(self, 'Webcam', 'No se pudo abrir la cámara.')
            return
        self.lastDetections = []
        self.timer.start(30)
        self.webcamButton.setText('Detener webcam')
        setVariant(self.webcamButton, 'danger')
        self.setStatus('● En vivo', 'ok')

    def stop(self):
        if not self.timer.isActive():
            return
        # stop timer and release video capture
        self.timer.stop()
        self.cap.release()
        self.cap = None
        self.webcamButton.setText('Iniciar webcam')
        setVariant(self.webcamButton, 'primary')
        self.setStatus('')
        self.video.clear()

    def shutdown(self):
        self.stop()
        # a QThread must not be destroyed while it is still running
        if self.worker is not None:
            self.worker.wait()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    theme.apply(app)
    page = FacePage()
    page.resize(1100, 700)
    page.show()
    sys.exit(app.exec())
