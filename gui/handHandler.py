# import system module
import sys
import math

# import some Qt modules
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout,
                               QLabel, QPushButton, QMessageBox, QSlider)
from PySide6.QtCore import QTimer, Qt, QSize

# import Opencv modules
import cv2
import numpy as np

import theme
from widgets import Page, Card, VideoView, setVariant, setProp

# region of interest where the hand must be placed
ROI_TOP, ROI_BOTTOM, ROI_LEFT, ROI_RIGHT = 100, 300, 100, 300

# range of skin color in YCrCb: it ignores the brightness (Y), so it is less sensitive to light than HSV
LOWER_SKIN = np.array([0, 135, 85], dtype=np.uint8)
UPPER_SKIN = np.array([255, 180, 135], dtype=np.uint8)
# looser range used together with the background, which already removes most of the false positives
LOWER_SKIN_LOOSE = np.array([0, 128, 70], dtype=np.uint8)
UPPER_SKIN_LOOSE = np.array([255, 190, 145], dtype=np.uint8)

OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
SHADOW_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# frames averaged to build the background, and how fast it adapts to light changes
CALIBRATION_FRAMES = 30
BACKGROUND_LEARNING_RATE = 0.02
DEFAULT_SENSITIVITY = 20

# colors in BGR (teal and peach from the theme)
ROI_COLOR = (201, 181, 31)
CALIBRATING_COLOR = (155, 197, 246)
HULL_COLOR = (155, 197, 246)
DEFECT_COLOR = (108, 111, 239)

# gesture id -> text shown in the interface
GESTURES = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
    'ok': 'OK', 'Best of luck': 'Suerte',
}


class HandSegmenter:
    """Separates the hand from the background inside the ROI.

    Without calibration it only uses the skin color, which fails when the background has a
    skin-like color (wood, orange walls, warm light). After calibrating with the empty ROI,
    the hand is what changed against that background AND has skin color.
    """

    def __init__(self):
        self.background = None
        self.samples = None
        # minimum difference (0-255) against the background to consider a pixel part of the hand
        self.threshold = DEFAULT_SENSITIVITY

    @property
    def calibrating(self):
        return self.samples is not None

    @property
    def calibrated(self):
        return self.background is not None

    def startCalibration(self):
        self.background = None
        self.samples = []

    def reset(self):
        self.background = None
        self.samples = None

    def segment(self, roi):
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        if self.calibrating:
            self.samples.append(blurred.astype(np.float32))
            if len(self.samples) >= CALIBRATION_FRAMES:
                self.background = np.mean(self.samples, axis=0)
                self.samples = None
            return np.zeros(roi.shape[:2], np.uint8)

        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        if not self.calibrated:
            return cleanMask(cv2.inRange(ycrcb, LOWER_SKIN, UPPER_SKIN))

        # pixels that changed against the background (largest difference of the 3 channels)
        current = blurred.astype(np.float32)
        diff = cv2.absdiff(current, self.background).max(axis=2)
        foreground = np.where(diff > self.threshold, 255, 0).astype(np.uint8)
        skin = cv2.inRange(ycrcb, LOWER_SKIN_LOOSE, UPPER_SKIN_LOOSE)

        # adapt slowly to light changes, only where there is no hand
        still = cv2.bitwise_not(cv2.dilate(foreground, SHADOW_KERNEL))
        cv2.accumulateWeighted(current, self.background, BACKGROUND_LEARNING_RATE, mask=still)

        return cleanMask(cv2.bitwise_and(foreground, skin))


def cleanMask(mask):
    # remove isolated noise, then fill the small holes inside the hand
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
    # smooth the borders
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def countFingerGaps(approx, roi):
    # find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    if defects is None:
        return 0

    gaps = 0
    # OpenCV 4 returns shape (N, 1, 4) and OpenCV 5 returns (N, 4)
    for s, e, f, _ in defects.reshape(-1, 4):
        start = tuple(int(v) for v in approx[s][0])
        end = tuple(int(v) for v in approx[e][0])
        far = tuple(int(v) for v in approx[f][0])

        # find length of all sides of triangle
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)
        if a == 0 or b == 0 or c == 0:
            continue

        # area of the triangle with Heron's formula
        p = (a + b + c) / 2
        ar = math.sqrt(max(p * (p - a) * (p - b) * (p - c), 0))

        # distance between point and convex hull
        d = (2 * ar) / a

        # apply cosine rule here (clamped to avoid float errors outside [-1, 1])
        cosine = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d > 30:
            gaps += 1
            cv2.circle(roi, far, 4, DEFECT_COLOR, -1)

        # draw lines around hand
        cv2.line(roi, start, end, HULL_COLOR, 2)

    return gaps


def recognizeGesture(frame, segmenter):
    """Detects the gesture inside the ROI and draws the analysis over the frame (in place).

    Returns (gesture, fingers, mask). gesture is None when there is no hand in the box,
    'reposition' when the shape is not a known gesture, or one of the GESTURES keys.
    """
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

    # the mask is computed before drawing anything over the frame
    mask = segmenter.segment(roi)

    # the box is drawn just outside the ROI so it is never part of the analyzed pixels
    color = CALIBRATING_COLOR if segmenter.calibrating else ROI_COLOR
    cv2.rectangle(frame, (ROI_LEFT - 2, ROI_TOP - 2), (ROI_RIGHT + 1, ROI_BOTTOM + 1), color, 2)

    # find contour of max area(hand)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask
    cnt = max(contours, key=cv2.contourArea)

    # define area of hull and area of hand
    areacnt = cv2.contourArea(cnt)
    if areacnt < 2000:
        return None, None, mask
    areahull = cv2.contourArea(cv2.convexHull(cnt))

    # find the percentage of area not covered by hand in convex hull
    arearatio = ((areahull - areacnt) / areacnt) * 100

    # approx the contour a little
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # number of fingers = gaps between fingers + 1
    fingers = countFingerGaps(approx, roi) + 1

    # choose the gesture which is in its range
    if fingers == 1:
        if arearatio < 12:
            gesture = '0'
        elif arearatio < 17.5:
            gesture = 'Best of luck'
        else:
            gesture = '1'
    elif fingers == 3:
        gesture = '3' if arearatio < 27 else 'ok'
    elif fingers in (2, 4, 5):
        gesture = str(fingers)
    else:
        gesture = 'reposition'

    return gesture, fingers, mask


class HandPage(Page):
    def __init__(self, parent=None):
        super().__init__('Gestos de la mano',
                         'Coloca la mano dentro del recuadro turquesa y muestra un gesto.', parent)
        self.cap = None
        self.segmenter = HandSegmenter()

        content = QHBoxLayout()
        content.setSpacing(20)
        self.body.addLayout(content, 1)

        main = QVBoxLayout()
        main.setSpacing(16)
        content.addLayout(main, 1)

        self.video = VideoView('Cámara apagada\n\nPulsa «Iniciar cámara» para comenzar')
        self.video.setMinimumSize(QSize(360, 270))
        main.addWidget(self.video, 1)
        main.addWidget(self.buildCalibrationBar())

        side = QVBoxLayout()
        side.setSpacing(16)
        content.addLayout(side)

        # detected gesture, the recognizable gestures are below and the current one is highlighted
        resultCard = Card('Gesto detectado')
        resultCard.setFixedWidth(290)
        self.value = QLabel('—')
        self.value.setObjectName('bigValue')
        self.value.setAlignment(Qt.AlignCenter)
        self.detail = QLabel('Esperando imagen')
        self.detail.setObjectName('muted')
        self.detail.setAlignment(Qt.AlignCenter)
        resultCard.body.addWidget(self.value)
        resultCard.body.addWidget(self.detail)
        resultCard.body.addSpacing(6)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.chips = {}
        for i, (gesture, text) in enumerate(GESTURES.items()):
            chip = QLabel(text)
            chip.setObjectName('chip')
            chip.setAlignment(Qt.AlignCenter)
            grid.addWidget(chip, i // 4, i % 4)
            self.chips[gesture] = chip
        resultCard.body.addLayout(grid)
        side.addWidget(resultCard)

        # hand mask, useful to understand what the algorithm sees
        maskCard = Card('Máscara de la mano')
        maskCard.setFixedWidth(290)
        self.maskView = VideoView('—', radius=10)
        self.maskView.setMinimumHeight(120)
        maskCard.body.addWidget(self.maskView, 1)
        side.addWidget(maskCard, 1)

        self.controlButton = QPushButton('Iniciar cámara')
        self.controlButton.setObjectName('primary')
        self.controlButton.setCursor(Qt.PointingHandCursor)
        self.controlButton.clicked.connect(self.controlTimer)
        side.addWidget(self.controlButton)

        # create a timer, each tick processes a single frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.getHandRec)

        # gives the user time to take the hand out of the box before calibrating
        self.calibrationDelay = QTimer(self)
        self.calibrationDelay.setSingleShot(True)
        self.calibrationDelay.timeout.connect(self.segmenter.startCalibration)

        self.updateCalibrationState()

    def buildCalibrationBar(self):
        bar = Card()
        bar.body.setContentsMargins(14, 10, 18, 10)
        row = QHBoxLayout()
        row.setSpacing(14)
        bar.body.addLayout(row)

        self.calibrateButton = QPushButton('Calibrar fondo')
        self.calibrateButton.setObjectName('secondary')
        self.calibrateButton.setCursor(Qt.PointingHandCursor)
        self.calibrateButton.setToolTip('Con la mano fuera del recuadro, guarda cómo se ve el fondo.\n'
                                        'Así la mano se detecta por lo que cambia y no solo por su color.')
        self.calibrateButton.clicked.connect(self.calibrate)
        row.addWidget(self.calibrateButton)
        row.addStretch()

        thresholdText = QLabel('Umbral')
        thresholdText.setObjectName('muted')
        row.addWidget(thresholdText)

        self.thresholdSlider = QSlider(Qt.Horizontal)
        self.thresholdSlider.setRange(5, 60)
        self.thresholdSlider.setValue(DEFAULT_SENSITIVITY)
        self.thresholdSlider.setFixedWidth(130)
        self.thresholdSlider.setToolTip('Diferencia mínima contra el fondo calibrado.\n'
                                        'Súbelo si aparece ruido en la máscara, bájalo si se pierden dedos.')
        self.thresholdSlider.valueChanged.connect(self.setThreshold)
        row.addWidget(self.thresholdSlider)

        self.thresholdValue = QLabel(str(DEFAULT_SENSITIVITY))
        self.thresholdValue.setFixedWidth(22)
        row.addWidget(self.thresholdValue)
        return bar

    def setThreshold(self, value):
        self.segmenter.threshold = value
        self.thresholdValue.setText(str(value))

    def calibrate(self):
        self.segmenter.reset()
        self.calibrationDelay.start(1500)
        self.updateCalibrationState()

    def updateCalibrationState(self):
        # the calibration state is shown in the page header while the camera is on
        if not self.timer.isActive():
            text, kind = '', ''
        elif self.calibrationDelay.isActive():
            text, kind = '● En vivo · Saca la mano del recuadro…', 'warn'
        elif self.segmenter.calibrating:
            text, kind = '● En vivo · Calibrando fondo…', 'warn'
        elif self.segmenter.calibrated:
            text, kind = '● En vivo · Fondo calibrado', 'ok'
        else:
            text, kind = '● En vivo · Sin calibrar, solo color de piel', ''
        if self.status.text() != text:
            self.setStatus(text, kind)

        busy = self.calibrationDelay.isActive() or self.segmenter.calibrating
        self.calibrateButton.setEnabled(self.timer.isActive() and not busy)
        self.calibrateButton.setText('Recalibrar' if self.segmenter.calibrated else 'Calibrar fondo')

    def getHandRec(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)

        gesture, fingers, mask = recognizeGesture(frame, self.segmenter)
        if self.segmenter.calibrating or self.calibrationDelay.isActive():
            self.showGesture(None, None)
            self.detail.setText('Calibrando…')
        else:
            self.showGesture(gesture, fingers)
        self.updateCalibrationState()

        self.video.setFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.maskView.setFrame(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

    def showGesture(self, gesture, fingers):
        if gesture is None:
            text, detail = '—', 'Coloca la mano en el recuadro'
        elif gesture == 'reposition':
            text, detail = '?', 'Reacomoda la mano'
        else:
            text = GESTURES[gesture]
            detail = '1 dedo' if fingers == 1 else f'{fingers} dedos'

        self.value.setText(text)
        setProp(self.value, 'long', len(text) > 2)
        self.detail.setText(detail)
        for key, chip in self.chips.items():
            active = key == gesture
            if chip.property('active') != active:
                setProp(chip, 'active', active)

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
        self.timer.start(30)
        self.controlButton.setText('Detener cámara')
        setVariant(self.controlButton, 'danger')
        self.updateCalibrationState()

    def stop(self):
        # stop timer and release video capture
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        # the camera may move between sessions, so the background must be calibrated again
        self.calibrationDelay.stop()
        self.segmenter.reset()
        self.updateCalibrationState()
        self.controlButton.setText('Iniciar cámara')
        setVariant(self.controlButton, 'primary')
        self.setStatus('')
        self.video.clear()
        self.maskView.clear()
        self.showGesture(None, None)
        self.detail.setText('Esperando imagen')

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    theme.apply(app)
    page = HandPage()
    page.resize(1100, 700)
    page.show()
    sys.exit(app.exec())
