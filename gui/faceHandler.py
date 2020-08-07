# import system module
import sys
import os
from os import path

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

# import Opencv modules
import face_recognition as faces
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faceGUI import *

#create array of encodings and names
know_encode = []
know_names = []

# for live recognition
live_face_location = []
live_face_encode = []
face_names = []

#the grasita function
def loadKnownFaces():
    obj = os.scandir('./img/known')
    for entry in obj:
        n = entry.name
        know_names.append(n[:-4])
        img = faces.load_image_file('./img/known/'+n)
        i_encode = faces.face_encodings(img)[0]
        know_encode.append(i_encode)
        print(n[:-4] + ' proccess')

class MainWindow(QWidget):
     # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.ui.btnImage.clicked.connect(self.uploadImage)
        loadKnownFaces()

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.WebCamRec)
        # set control_bt callback clicked  function
        self.ui.btnWebcam.clicked.connect(self.controlTimer)
    
    def uploadImage(self):
        self.fileDialog = QFileDialog(self)
        fname = self.fileDialog.getOpenFileUrl()
        imgf = fname[0].path()
        self.ImageIdentify(imgf[1:])

    def ImageIdentify(self, pathImg):
        #load image
        upImg = faces.load_image_file(pathImg)

        # find faces in images
        face_location = faces.face_locations(upImg)
        face_encode = faces.face_encodings(upImg, face_location)

        # convert to PIL format
        pil_img = Image.fromarray(upImg)

        #create a draw instance
        drawImg = ImageDraw.Draw(pil_img)

        #loop the faces in images
        for(top, right, bottom, left), face_encode in zip(face_location, face_encode):
            matches = faces.compare_faces(know_encode, face_encode)
            name = "Unknown Person"

            if True in matches:
                first_match_index = matches.index(True)
                name = know_names[first_match_index]

            #draw box
            drawImg.rectangle(((left, top), (right, bottom)), outline=(0,0,0))

            #draw label
            font = ImageFont.truetype("arial.ttf", 16)
            text_width, text_height = drawImg.textsize(name)
            drawImg.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
            drawImg.text((left+6, bottom - text_height - 5), name, font=font, fill=(255,255,255,255))

        #create scanned-images if necessary
        if not os.path.exists('.\scannedImages'):
            os.mkdir('.\scannedImages')

        #show image
        pathTmpImg = '.\scannedImages\TEMP_Img.PNG'
        pil_img.save(pathTmpImg)
        os.system(pathTmpImg)

    def WebCamRec(self):
        self.process_this_frame = True
        while self.flowControl:
            # Grab a single frame of video
            ret, frame = self.cap.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which faces uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if self.process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                live_face_location = faces.face_locations(rgb_small_frame)
                live_face_encode = faces.face_encodings(rgb_small_frame, live_face_location)

                face_names = []
                for face_encoding in live_face_encode:
                    # See if the face is a match for the known face(s)
                    matches = faces.compare_faces(know_encode, face_encoding)
                    name = "Unknown"

                    # # If a match was found in know_encode, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = know_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = faces.face_distance(know_encode, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = know_names[best_match_index]

                    face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(live_face_location, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            # cv2.imshow('Video', frame)
            # convert image to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # get image infos
            height, width, channel = frame.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.lbl_tempImg.setPixmap(QPixmap.fromImage(qImg))

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(0)
            # start timer
            self.timer.start(20)
            self.flowControl = True
            # update control_bt text
            self.ui.btnWebcam.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            self.flowControl = False
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.btnWebcam.setText("Scan Webcam")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = MainWindow()
    GUI.show()
    sys.exit(app.exec_())