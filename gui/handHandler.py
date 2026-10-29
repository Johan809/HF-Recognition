# import system module
import sys
import math
import random
from collections import namedtuple, deque, Counter

# import some Qt modules
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout,
                               QLabel, QPushButton, QMessageBox, QSlider)
from PySide6.QtCore import QTimer, Qt, QSize, QSettings
from PySide6.QtGui import QImage

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
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SHADOW_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# frames averaged to build the background, and how fast it adapts to light changes
CALIBRATION_FRAMES = 30
BACKGROUND_LEARNING_RATE = 0.02
DEFAULT_SENSITIVITY = 30
MAX_SENSITIVITY = 100

# fingers are searched on circles around the palm, with a radius relative to the palm size,
# so the result does not depend on how far the hand is from the camera
SAMPLING_RADII = (1.6, 1.9, 2.2, 2.5)
SAMPLES_PER_CIRCLE = 360
# arcs pointing down are the wrist/forearm entering the box (down is +90° in image coordinates)
WRIST_SECTOR = math.radians(55)
# a thumb up must point less than this away from straight up
THUMB_UP_SECTOR = math.radians(60)
# widths relative to the palm radius
MIN_FINGER_WIDTH = 0.15
FINGER_WIDTH = 0.45
MERGED_FINGERS_WIDTH = 0.95
THUMB_WIDTH = 0.6
MAX_THUMB_WIDTH = 1.05
# how far (in palm radii) the hand must go on beyond a circle for an arc to be a finger
FINGER_EXTENSION = 0.15
MIN_PALM_RADIUS = 8
MIN_HAND_AREA = 2000
# frames used to stabilize the shown gesture (about 0.25 s)
SMOOTHING_FRAMES = 8

# fun mode: gesture id -> cat images in img/cat_gestures (one is chosen at random)
CAT_DIR = theme.ROOT / 'img' / 'cat_gestures'
CAT_IMAGES = {
    '0': ['fist'], '1': ['1'], '2': ['2', 'peace-sign'], '3': ['3'], '4': ['4'], '5': ['5'], 'ok': ['ok'],
}

# colors in BGR (teal and peach from the theme)
ROI_COLOR = (201, 181, 31)
CALIBRATING_COLOR = (155, 197, 246)
HULL_COLOR = (155, 197, 246)
PALM_COLOR = (243, 237, 230)
TIP_COLOR = (108, 111, 239)

# gesture id -> text shown in the interface
GESTURES = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', 'ok': 'OK',
}

Finger = namedtuple('Finger', 'angle width tip')


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

        # webcams change the exposure and white balance on their own (for example when the hand
        # enters), which makes the whole box look different; estimate that change per channel with
        # the median ratio against the background (the hand covers less than half of the box)
        current = blurred.astype(np.float32)
        ratio = current[::4, ::4] / np.maximum(self.background[::4, ::4], 1)
        gain = np.clip(np.median(ratio.reshape(-1, 3), axis=0), 0.5, 2.0)

        # pixels that changed against the background (largest difference of the 3 channels)
        diff = cv2.absdiff(current, self.background * gain).max(axis=2)
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


def palmCircle(hand):
    """Center and radius of the biggest circle that fits inside the hand (the palm, or the fist)."""
    # the borders of the box count as background, so the forearm (which enters from the bottom
    # and is cut by the box) can not hold a bigger circle than the palm
    padded = cv2.copyMakeBorder(hand, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return (int(x), int(y)), float(dist[y, x])


def fingersOnCircle(hand, center, radius, palmRadius):
    """Fingers crossing a circle around the palm: each run of hand pixels along the circle."""
    angles = np.linspace(-math.pi, math.pi, SAMPLES_PER_CIRCLE, endpoint=False)
    xs = np.round(center[0] + radius * np.cos(angles)).astype(int)
    ys = np.round(center[1] + radius * np.sin(angles)).astype(int)
    height, width = hand.shape
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    on = np.zeros(SAMPLES_PER_CIRCLE, bool)
    on[inside] = hand[ys[inside], xs[inside]] > 0
    if on.all() or not on.any():
        return []

    # start on a background sample, so no run is split between the end and the start of the array
    start = int(np.argmin(on))
    on = np.roll(on, -start)
    step = 2 * math.pi / SAMPLES_PER_CIRCLE

    fingers = []
    i = 0
    while i < SAMPLES_PER_CIRCLE:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < SAMPLES_PER_CIRCLE and on[j]:
            j += 1
        # direction of the middle of the run, and its width (arc length) relative to the palm
        angle = float(angles[(start + (i + j - 1) // 2) % SAMPLES_PER_CIRCLE])
        arcWidth = (j - i) * step * radius / palmRadius
        i = j

        # ignore noise, the wrist, and the corners of the fist (the hand must go on beyond the circle)
        if arcWidth < MIN_FINGER_WIDTH or abs(angle - math.pi / 2) < WRIST_SECTOR:
            continue
        if not isHand(hand, center, radius + FINGER_EXTENSION * palmRadius, angle):
            continue
        tip = (int(center[0] + radius * math.cos(angle)), int(center[1] + radius * math.sin(angle)))
        fingers.append(Finger(angle, arcWidth, tip))
    return fingers


def isHand(hand, center, radius, angle):
    x = int(round(center[0] + radius * math.cos(angle)))
    y = int(round(center[1] + radius * math.sin(angle)))
    height, width = hand.shape
    # a finger that leaves the box also goes on
    if not (0 <= x < width and 0 <= y < height):
        return True
    return hand[y, x] > 0


def findFingers(hand, center, palmRadius):
    """Arcs where the fingers cross the circle that shows the most separate fingers.

    Fingers spread out towards the tips, so an outer circle separates fingers that are still
    together near the palm; short fingers (thumb, pinky) are found by the inner circles.
    """
    best = []
    for k in SAMPLING_RADII:
        arcs = fingersOnCircle(hand, center, k * palmRadius, palmRadius)
        if len(arcs) > len(best):
            best = arcs
    return best


def countFingers(arcs):
    # fingers held together cross even the best circle as a single wide arc
    return sum(max(1, round(arc.width / FINGER_WIDTH)) if arc.width > MERGED_FINGERS_WIDTH else 1
               for arc in arcs)


def isThumbUp(arcs):
    # a single protrusion pointing up, much wider than a finger compared to the palm: seen from the
    # side the fist is narrow (small palm radius) and the thumb is the thickest finger. Two fingers
    # held together would be as wide, but they almost always separate on one of the circles.
    if len(arcs) != 1:
        return False
    arc = arcs[0]
    pointsUp = abs(arc.angle + math.pi / 2) < THUMB_UP_SECTOR
    return pointsUp and THUMB_WIDTH < arc.width < MAX_THUMB_WIDTH


def analyzeHand(mask, roi):
    """Counts the fingers of the biggest blob of the mask and draws the analysis over the roi.

    Returns (gesture, fingers). gesture is None when there is no hand, 'reposition' when the
    shape is not a known gesture, or one of the GESTURES keys.
    """
    # find contour of max area(hand)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_HAND_AREA:
        return None, None

    # measure only the hand, without the small blobs of noise
    hand = np.zeros_like(mask)
    cv2.drawContours(hand, [cnt], -1, 255, cv2.FILLED)
    center, palmRadius = palmCircle(hand)
    if palmRadius < MIN_PALM_RADIUS:
        return None, None
    arcs = findFingers(hand, center, palmRadius)

    # draw the convex hull, the palm and a line to each finger
    cv2.drawContours(roi, [cv2.convexHull(cnt)], -1, HULL_COLOR, 1)
    cv2.circle(roi, center, int(palmRadius), PALM_COLOR, 1)
    for arc in arcs:
        cv2.line(roi, center, arc.tip, PALM_COLOR, 1)
        cv2.circle(roi, arc.tip, 4, TIP_COLOR, -1)

    if isThumbUp(arcs):
        return 'ok', 1
    count = countFingers(arcs)
    if count <= 5:
        return str(count), count
    return 'reposition', count


class GestureSmoother:
    """Changes the shown gesture only when the new one is the majority of the last frames,
    so a single bad frame does not make it flicker."""

    def __init__(self, frames=SMOOTHING_FRAMES):
        self.history = deque(maxlen=frames)
        self.current = (None, None)

    def update(self, gesture, fingers):
        self.history.append((gesture, fingers))
        best, votes = Counter(self.history).most_common(1)[0]
        if votes > len(self.history) // 2:
            self.current = best
        return self.current

    def reset(self):
        self.history.clear()
        self.current = (None, None)


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

    gesture, fingers = analyzeHand(mask, roi)
    return gesture, fingers, mask


def loadCatImages():
    """gesture id -> list of QImage, only for the images that exist in img/cat_gestures."""
    images = {}
    for gesture, names in CAT_IMAGES.items():
        found = [QImage(str(path)) for name in names for path in sorted(CAT_DIR.glob(name + '.*'))]
        found = [image for image in found if not image.isNull()]
        if found:
            images[gesture] = found
    return images


class HandPage(Page):
    def __init__(self, parent=None):
        super().__init__('Gestos de la mano',
                         'Coloca la mano dentro del recuadro turquesa y muestra un gesto.', parent)
        self.cap = None
        self.segmenter = HandSegmenter()
        self.smoother = GestureSmoother()
        self.settings = QSettings('HF Recognition', 'HF Recognition')
        self.catImages = loadCatImages()
        self.catGesture = None

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

        self.side = side = QVBoxLayout()
        side.setSpacing(16)
        content.addLayout(side)

        # detected gesture, the recognizable gestures are below and the current one is highlighted
        self.resultCard = resultCard = Card()
        resultCard.setFixedWidth(290)
        header = QHBoxLayout()
        title = QLabel('GESTO DETECTADO')
        title.setObjectName('cardTitle')
        header.addWidget(title)
        header.addStretch()

        # fun mode: shows the cat of the gesture instead of the number
        self.funButton = QPushButton('Modo gato')
        self.funButton.setObjectName('toggle')
        self.funButton.setCheckable(True)
        self.funButton.setCursor(Qt.PointingHandCursor)
        if self.catImages:
            self.funButton.setToolTip('Muestra el gato del gesto en vez del número')
        else:
            self.funButton.setEnabled(False)
            self.funButton.setToolTip('No hay imágenes en img/cat_gestures')
        self.funButton.toggled.connect(self.setFunMode)
        header.addWidget(self.funButton)
        resultCard.body.addLayout(header)

        self.value = QLabel('—')
        self.value.setObjectName('bigValue')
        self.value.setAlignment(Qt.AlignCenter)
        self.catView = VideoView('Muestra un gesto', radius=10)
        self.catView.setMinimumHeight(110)
        self.catView.hide()
        self.detail = QLabel('Esperando imagen')
        self.detail.setObjectName('muted')
        self.detail.setAlignment(Qt.AlignCenter)
        resultCard.body.addWidget(self.value)
        resultCard.body.addWidget(self.catView, 1)
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
        # the fun mode is remembered between sessions
        self.funButton.setChecked(bool(self.catImages) and self.settings.value('funMode', False, type=bool))

    def setFunMode(self, on):
        self.value.setVisible(not on)
        self.catView.setVisible(on)
        # the cat image takes most of the free space of the panel, the number does not need it
        self.side.setStretchFactor(self.resultCard, 2 if on else 0)
        self.settings.setValue('funMode', on)

    def updateCatView(self, gesture):
        if gesture == self.catGesture:
            return
        self.catGesture = gesture
        images = self.catImages.get(gesture)
        if images:
            self.catView.setImage(random.choice(images))
        else:
            self.catView.placeholder = 'Reacomoda la mano' if gesture == 'reposition' else 'Muestra un gesto'
            self.catView.clear()

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
        self.thresholdSlider.setRange(5, MAX_SENSITIVITY)
        self.thresholdSlider.setValue(DEFAULT_SENSITIVITY)
        self.thresholdSlider.setFixedWidth(130)
        self.thresholdSlider.setToolTip('Diferencia mínima contra el fondo calibrado.\n'
                                        'Súbelo si aparece ruido en la máscara, bájalo si se pierden dedos.')
        self.thresholdSlider.valueChanged.connect(self.setThreshold)
        row.addWidget(self.thresholdSlider)

        self.thresholdValue = QLabel(str(DEFAULT_SENSITIVITY))
        self.thresholdValue.setFixedWidth(28)
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
            self.smoother.reset()
            self.showGesture(None, None)
            self.detail.setText('Calibrando…')
        else:
            self.showGesture(*self.smoother.update(gesture, fingers))
        self.updateCalibrationState()

        self.video.setFrame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.maskView.setFrame(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))

    def showGesture(self, gesture, fingers):
        if gesture is None:
            text, detail = '—', 'Coloca la mano en el recuadro'
        elif gesture == 'reposition':
            text, detail = '?', 'Reacomoda la mano'
        elif gesture == 'ok':
            text, detail = GESTURES[gesture], 'Pulgar arriba'
        else:
            text = GESTURES[gesture]
            detail = '1 dedo' if fingers == 1 else f'{fingers} dedos'

        self.value.setText(text)
        setProp(self.value, 'long', len(text) > 2)
        self.detail.setText(detail)
        self.updateCatView(gesture)
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
        self.smoother.reset()
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
