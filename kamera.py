import sys
import tensorflow as tf
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

class VideoCaptureWidget(QWidget):
    def __init__(self, model, face_cascade):
        super().__init__()
        self.model = model
        self.face_cascade = face_cascade
        self.bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Dodavanje bins ovde
        self.initUI()

    def initUI(self):
        self.video_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            self.timer.stop()
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            preprocessed_face = cv2.resize(face, (150, 150))
            preprocessed_face = img_to_array(preprocessed_face)
            preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
            preprocessed_face = preprocess_input(preprocessed_face)
            age_prediction = self.model.predict(preprocessed_face)
            predicted_bin = np.argmax(age_prediction)
            predicted_age_interval = self.bins[predicted_bin:predicted_bin + 2]
            predicted_age_text = f"{predicted_age_interval[0]}-{predicted_age_interval[1]}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {predicted_age_text}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.capture.release()

def main():
    app = QApplication(sys.argv)

    model = tf.keras.models.load_model('noviModel.h5')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    video_capture_widget = VideoCaptureWidget(model, face_cascade)
    video_capture_widget.setWindowTitle('Age Estimation')
    video_capture_widget.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
