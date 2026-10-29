# Console version of the hand gesture recognizer (no Qt window, only OpenCV)
# Keys: 'c' calibrates the background (take the hand out of the box first), 'q' quits
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gui'))
from handHandler import recognizeGesture, HandSegmenter

cap = cv2.VideoCapture(0)
segmenter = HandSegmenter()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gesture, fingers, mask = recognizeGesture(frame, segmenter)
    if segmenter.calibrating:
        text = 'Calibrating...'
    else:
        text = 'Put hand in the box' if gesture is None else gesture
    cv2.putText(frame, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

    # show the windows
    cv2.imshow('mask', mask)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        segmenter.startCalibration()
    # Hit 'q' on the keyboard to quit!
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
