from pathlib import Path

from PySide6.QtGui import QFont

# project folders, resolved from this file so the app runs from any working directory
ROOT = Path(__file__).resolve().parent.parent
VISUAL_DIR = ROOT / 'gui' / 'visual'
KNOWN_DIR = ROOT / 'img' / 'known'
SCANNED_DIR = ROOT / 'scannedImages'

# palette taken from the original visual design (navy, teal and peach)
BG = '#11161c'
SURFACE = '#19212a'
SURFACE_ALT = '#212b36'
VIDEO_BG = '#0b0f13'
BORDER = '#2a3542'
TEXT = '#e6edf3'
MUTED = '#8796a5'
ACCENT = '#1fb5c9'
ACCENT_HOVER = '#45c9da'
ACCENT_SOFT = 'rgba(31, 181, 201, 0.14)'
PEACH = '#f6c59b'
DANGER = '#ef6f6c'
DANGER_HOVER = '#f38987'

QSS = """
QMainWindow, #page { background: @BG; }
QLabel { background: transparent; color: @TEXT; }
QToolTip { background: @SURFACE_ALT; color: @TEXT; border: 1px solid @BORDER; padding: 6px; }

#sidebar { background: @SURFACE; border-right: 1px solid @BORDER; }
#brandName { font-size: 17px; font-weight: 700; }
#brandSub, #footnote { color: @MUTED; font-size: 12px; }

QPushButton[nav="true"] {
    text-align: left; padding: 11px 14px; border: none; border-radius: 9px;
    background: transparent; color: @MUTED; font-size: 14px; font-weight: 600;
}
QPushButton[nav="true"]:hover { background: @SURFACE_ALT; color: @TEXT; }
QPushButton[nav="true"]:checked { background: @ACCENT_SOFT; color: @ACCENT; }

QPushButton#primary, QPushButton#danger, QPushButton#secondary {
    padding: 11px 18px; border-radius: 9px; font-size: 14px; font-weight: 600;
}
QPushButton#primary { background: @ACCENT; color: #062027; border: none; }
QPushButton#primary:hover { background: @ACCENT_HOVER; }
QPushButton#danger { background: @DANGER; color: #2a0b0a; border: none; }
QPushButton#danger:hover { background: @DANGER_HOVER; }
QPushButton#secondary { background: transparent; color: @TEXT; border: 1px solid @BORDER; }
QPushButton#secondary:hover { border-color: @ACCENT; color: @ACCENT; }
QPushButton:disabled { background: @SURFACE_ALT; color: @MUTED; border: none; }

QPushButton#link {
    background: transparent; border: none; color: @ACCENT; font-weight: 600; padding: 2px 0;
}
QPushButton#link:hover { color: @ACCENT_HOVER; text-decoration: underline; }

#card { background: @SURFACE; border: 1px solid @BORDER; border-radius: 14px; }
#cardTitle { color: @MUTED; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
#pageTitle { font-size: 26px; font-weight: 700; }
#pageSubtitle, #muted { color: @MUTED; }
#sectionTitle { font-size: 16px; font-weight: 700; }
#featureTitle { font-size: 18px; font-weight: 700; }
#featureText { color: @MUTED; line-height: 140%; }

#bigValue { color: @PEACH; font-size: 64px; font-weight: 700; }
#bigValue[long="true"] { font-size: 30px; }

#chip {
    background: @SURFACE_ALT; color: @MUTED; border: 1px solid transparent;
    border-radius: 8px; padding: 6px 10px; font-weight: 600;
}
#chip[active="true"] { background: @ACCENT_SOFT; color: @ACCENT; border-color: @ACCENT; }

#tag {
    background: @SURFACE_ALT; color: @PEACH; border-radius: 6px; padding: 3px 8px;
    font-size: 12px; font-weight: 600;
}

#status { color: @MUTED; font-size: 13px; }
#status[kind="ok"] { color: @ACCENT; }
#status[kind="warn"] { color: @PEACH; }

QSlider::groove:horizontal { height: 4px; background: @BORDER; border-radius: 2px; }
QSlider::sub-page:horizontal { background: @ACCENT; border-radius: 2px; }
QSlider::handle:horizontal {
    background: @TEXT; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: @ACCENT_HOVER; }

QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 6px 4px; border-radius: 8px; color: @TEXT; }
QListWidget::item:hover { background: @SURFACE_ALT; }
QListWidget::item:selected { background: @ACCENT_SOFT; color: @ACCENT; }

QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: @BORDER; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QMessageBox { background: @SURFACE; }
QMessageBox QLabel { color: @TEXT; }
"""


def styleSheet():
    tokens = {
        'BG': BG, 'SURFACE_ALT': SURFACE_ALT, 'SURFACE': SURFACE, 'BORDER': BORDER,
        'TEXT': TEXT, 'MUTED': MUTED, 'ACCENT_HOVER': ACCENT_HOVER, 'ACCENT_SOFT': ACCENT_SOFT,
        'ACCENT': ACCENT, 'PEACH': PEACH, 'DANGER_HOVER': DANGER_HOVER, 'DANGER': DANGER,
    }
    qss = QSS
    # longest names first so @SURFACE does not eat @SURFACE_ALT
    for name in sorted(tokens, key=len, reverse=True):
        qss = qss.replace('@' + name, tokens[name])
    return qss


def apply(app):
    app.setStyle('Fusion')
    font = QFont()
    font.setFamilies(['Segoe UI Variable Text', 'Segoe UI', 'Inter', 'Helvetica Neue', 'Arial'])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(styleSheet())
