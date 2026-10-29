from PySide6.QtCore import Qt, QRectF, QPointF, QSizeF, QThread, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QColor, QPen, QPixmap
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy

import theme


def repolish(widget):
    # re-apply the stylesheet after changing an objectName or a dynamic property
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def setVariant(button, variant):
    button.setObjectName(variant)
    repolish(button)


def setProp(widget, name, value):
    widget.setProperty(name, value)
    repolish(widget)


def rgbToImage(rgb):
    height, width, channel = rgb.shape
    return QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888).copy()


def roundedPixmap(pixmap, size, radius):
    """Scales a pixmap to cover `size` and clips it with rounded corners."""
    scaled = pixmap.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    result = QPixmap(size)
    result.fill(Qt.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, size.width(), size.height()), radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap((size.width() - scaled.width()) // 2, (size.height() - scaled.height()) // 2, scaled)
    painter.end()
    return result


class Page(QWidget):
    """Base class for the pages: dark background, header with title and subtitle."""

    def __init__(self, title, subtitle, parent=None):
        super().__init__(parent)
        self.setObjectName('page')
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(32, 28, 32, 28)
        self.body.setSpacing(20)

        header = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(4)
        titleLabel = QLabel(title)
        titleLabel.setObjectName('pageTitle')
        subtitleLabel = QLabel(subtitle)
        subtitleLabel.setObjectName('pageSubtitle')
        texts.addWidget(titleLabel)
        texts.addWidget(subtitleLabel)
        header.addLayout(texts)
        header.addStretch()

        self.status = QLabel('')
        self.status.setObjectName('status')
        header.addWidget(self.status, 0, Qt.AlignBottom)
        self.body.addLayout(header)

    def setStatus(self, text, kind=''):
        self.status.setText(text)
        setProp(self.status, 'kind', kind)

    def stop(self):
        # pages with a running camera override this
        pass

    def shutdown(self):
        # called when the application closes
        self.stop()


class Card(QFrame):
    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setObjectName('card')
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(10)
        if title:
            label = QLabel(title.upper())
            label.setObjectName('cardTitle')
            self.body.addWidget(label)


class VideoView(QWidget):
    """Shows a frame (RGB numpy array) scaled to fit, inside a rounded dark panel."""

    def __init__(self, placeholder='', radius=14, parent=None):
        super().__init__(parent)
        self.placeholder = placeholder
        self.radius = radius
        self.image = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setFrame(self, rgb):
        self.image = rgbToImage(rgb)
        self.update()

    def clear(self):
        self.image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(rect, self.radius, self.radius)
        painter.fillPath(path, QColor(theme.VIDEO_BG))

        if self.image is None:
            painter.setPen(QColor(theme.MUTED))
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.placeholder)
        else:
            size = QSizeF(self.image.size().scaled(self.size(), Qt.KeepAspectRatio))
            origin = QPointF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2)
            painter.setClipPath(path)
            painter.drawImage(QRectF(origin, size), self.image)
            painter.setClipping(False)

        painter.setPen(QPen(QColor(theme.BORDER), 1))
        painter.drawPath(path)


class Banner(QWidget):
    """Draws a pixmap covering the whole widget with rounded corners."""

    def __init__(self, pixmap, height, radius=16, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.radius = radius
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self.radius, self.radius)
        painter.setClipPath(path)
        scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        painter.drawPixmap((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled)


class Worker(QThread):
    """Runs a function in a background thread so the interface never freezes."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.args = args

    def run(self):
        try:
            self.done.emit(self.fn(*self.args))
        except Exception as error:
            self.failed.emit(str(error))
