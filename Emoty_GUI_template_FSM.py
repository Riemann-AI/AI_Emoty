from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

classifier = load_model('model_v6_23.hdf5')
class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# import Opencv module
import cv2

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# import generic modules
import sys
import numpy as np

# import some PyQt5 modules
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import (QApplication, QWidget)
from PyQt5.QtGui import (QImage, QPixmap)
from PyQt5.QtCore import (Qt, QTimer, QRunnable, 
                          pyqtSignal, pyqtSlot, 
                          QThread, QSize)


class CameraEngine(QThread):
    output = pyqtSignal(QImage)
    def __init__(self, parent = None):
        QThread.__init__(self, parent)
        self.exiting = False
        
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eyes_classifier = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        
    # view camera
    def AiCamera(self, cap):
        # read image in BGR format
        self.cap = cap
        self.start()

    def AiEngine(self):
        self.ui.labelCurrentEmotion.setText("Waiting for emotion ...")
        self.ui.ImageFrame.setStyleSheet("border: 5px solid white")
    
    def render(self, gameStatusIndicator):    
        self.gameStatus = gameStatusIndicator
        #self.start()
        
    def run(self):        
        # Note: This is never called directly. It is called by Qt once the
        # thread environment has been set up.
        while not self.exiting:
            # read image in BGR format
            ret, frame = self.cap.read()
            # convert image to RGB format
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            #-- Detect faces"
            faces = self.face_classifier.detectMultiScale(
                gray,
                scaleFactor=1.3, # decreases by 1.3 times
                minNeighbors=5) # 5 specifies the number of times scaling happens
            
            
            # For creating a rectangle around the image 
            if len(faces) == 0:
                print("Waiting for emotion ...")
            else:
                for (x,y,w,h) in faces:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                
            # get image infos
            height, width, channel = image.shape
            scaled_size = QSize(832, 624)
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            qImg = qImg.scaled(scaled_size, Qt.KeepAspectRatio)
            self.output.emit(qImg)
            print("emitting")

        

class UI(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super(UI, self).__init__()
        self.ui = uic.loadUi('mainUI.ui', self) # Load the .ui file and assign to self.ui       
        self.threadCamera = CameraEngine()

        # SETTINGS
        self.targetEmotion = "Sad"
        self.scoreVal = 0
        
        self.gameStatus = 0 # 1 per avviare la camera
        
        # SETTING GLOBAL VARIABLES + INITIALIZING INDICATORS
        self.isFinished = False
        self.isMatch = False
        
        self.ui.labelTargetEmotion.setText(self.targetEmotion)
        self.ui.labelCurrentEmotion.setText("...")
        
        self.ui.labelTotalScore.setText(str(self.scoreVal))
        self.ui.progressBar.hide()
        
        self.ui.ImageFrame.setStyleSheet("border: 5px solid white")
        self.ui.progressBar.setValue(100)
        
        self.threadCamera.finished.connect(self.updateUI)
        self.threadCamera.output.connect(self.addImage)
        
    # create a timer
        self.timer = QTimer()
        self.actionPlay.triggered.connect(self.initCamera)#controlTimer)
        self.actionExit.triggered.connect(self.onEnd)
        
    def initCamera(self):
        print("initCamera")
        # create video capture
        self.cap = cv2.VideoCapture(0)
        self.ui.progressBar.show()
        self.gameStatus = 1
        self.threadCamera.render(self.gameStatus)
        self.threadCamera.AiCamera(self.cap)

    def updateUI(self):
        print("finito")
        
    def addImage(self, qImg):
        # show image in img_label
        self.ui.ImageLabel.setPixmap(QPixmap.fromImage(qImg))

    # funzione per la visualizzazione del tempo di gioco
    def setProgressBar(self):
        count = self.ui.progressBar.value()
        if count > 0:
            count = count - 1
        else:
            self.isFinished = True
            self.onEnd()
        self.ui.progressBar.setValue(count)
    
    # funzione per la gestione dello score
    def setScore(self):
        currentScore = self.scoreVal
        if self.isMatch == True:
            self.scoreVal = currentScore + 1
            self.isMatch = False
            self.isFinished = True
            self.onEnd()
        self.ui.labelTotalScore.setText(str(self.scoreVal))

    # DEATH STATE
    # funzione per chiudere l'app
    def onEnd(self):
        print("closing")
        self.cap.release()
        self.timer.stop()
        app.quit()


if __name__ == '__main__':
    
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = UI()
    mainWindow.show()

    sys.exit(app.exec_())