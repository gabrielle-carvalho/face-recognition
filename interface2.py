import sys
import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap


# Load Dlib's face detector and recognition model
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def prepare_training_data(data_folder_path):
    labels = []  # Names of the people
    descriptors = []  # Face descriptors

    for label in os.listdir(data_folder_path):
        person_dir = os.path.join(data_folder_path, label)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = sp(gray, face)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                descriptors.append(np.array(face_descriptor))
                labels.append(label)

    return labels, descriptors


# Function to compute face descriptor
def compute_face_descriptor(frame):
    faces = detector(frame)
    if len(faces) > 0:
        shape = sp(frame, faces[0])
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        return np.array(face_descriptor)
    return None

# Function to match face descriptors (you should implement this based on your dataset)
# Function to match face descriptors
def match_face(face_descriptor, labels, descriptors):
    # Find the descriptor in your database that is closest to the detected face
    distances = [distance.euclidean(face_descriptor, descriptor) for descriptor in descriptors]
    min_distance = min(distances)
    if min_distance < 0.6:  # Threshold, can be adjusted
        index = distances.index(min_distance)
        return labels[index]
    return "Unknown"


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Face Recognition Software'
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 600
        self.frame_counter = 0  # Initialize frame counter
        self.recognized_faces = {}
        self.labels, self.descriptors = prepare_training_data("images")
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Main layout
        main_layout = QVBoxLayout()

        # Camera label
        self.camera_label = QLabel(self)
        main_layout.addWidget(self.camera_label)

        # Name input
        self.name_input = QLineEdit(self)
        main_layout.addWidget(self.name_input)

        # Buttons layout
        buttons_layout = QHBoxLayout()

        # Capture button
        self.capture_button = QPushButton('Capture', self)
        self.capture_button.clicked.connect(self.on_capture_click)
        buttons_layout.addWidget(self.capture_button)

        # Retrain button
        self.retrain_button = QPushButton('Retrain', self)
        self.retrain_button.clicked.connect(self.on_retrain_click)
        buttons_layout.addWidget(self.retrain_button)

        # Add buttons layout to main layout
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0/30)

    def update_frame(self):
        ret, frame = self.cap.read()
        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Perform face recognition every 10 frames
            self.frame_counter += 1
            if self.frame_counter % 1 == 0:
                new_labels = {}
                for face in faces:
                    shape = sp(gray, face)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    name = match_face(np.array(face_descriptor), self.labels, self.descriptors)
                    new_labels[(face.left(), face.top())] = name
                self.face_labels = new_labels
            else:
                # Use existing labels if not performing recognition
                for face in faces:
                    if (face.left(), face.top()) not in self.face_labels:
                        self.face_labels[(face.left(), face.top())] = "Unknown"

            # Draw rectangles and labels for each face
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = self.face_labels.get((x, y), "Unknown")
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            # Resize the image to fit the window
            frame = cv2.resize(frame, (self.width, int(self.width * frame.shape[0] / frame.shape[1])))
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888).rgbSwapped()
            self.camera_label.setPixmap(QPixmap.fromImage(image))


    @pyqtSlot()
    def on_capture_click(self):
        name = self.name_input.text()
        ret, frame = self.cap.read()
        if ret:
            # Here you can add code to process and save the frame with the associated name
            path = f"./images/{name}"
            if not os.path.exists(path):
                os.mkdir(path)
            picnumber = len(os.listdir(path))
            cv2.imwrite(f'{path}/{picnumber}.png', frame)

    @pyqtSlot()
    def on_retrain_click(self):
        # Add code for the retraining process here
        self.labels, self.descriptors = prepare_training_data("images")
        print("Retrained the model with updated images")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
