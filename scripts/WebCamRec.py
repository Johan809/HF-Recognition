# Console version of the webcam face recognizer (no Qt window, only OpenCV)
import sys
from pathlib import Path

import cv2
import face_recognition

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gui'))
from faceHandler import loadKnownFaces, identify, drawFaces

# Load the known faces from img/known
known, _ = loadKnownFaces()

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

detections = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small = cv2.resize(rgb, (0, 0), fx=0.25, fy=0.25)
        locations = face_recognition.face_locations(small)
        encodings = face_recognition.face_encodings(small, locations)

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        detections = [(tuple(v * 4 for v in loc), identify(known, enc)) for loc, enc in zip(locations, encodings)]

    process_this_frame = not process_this_frame

    # Display the resulting image
    cv2.imshow('Video', cv2.cvtColor(drawFaces(rgb, detections), cv2.COLOR_RGB2BGR))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
