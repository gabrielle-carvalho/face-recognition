#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

# Inicializa os detectores do dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

class PeopleRecognition(Node): # MODIFY NAME
    def __init__(self):
        super().__init__("people_recognition") # MODIFY NAME
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.get_logger().error("Failed to open camera.")
        self.create_subscription(String, 'speech_text', self.capture_callback, 10)
        self.get_logger().info("People Recognition Node started")

    def capture_callback(self, msg):
        name = msg.data  
        ret, frame = self.camera.read()
        if not ret:
            self.get_logger().error("Failed to capture image.")
            return

        path = f"./images/{name}"
        os.makedirs(path, exist_ok=True) 
        picnumber = len(os.listdir(path))
        cv2.imwrite(f'{path}/{picnumber}.png', frame)
        self.get_logger().info(f"Image saved as {path}/{picnumber}.png")
        messagebox.showinfo("Info", "Image Captured Successfully!")

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
            if img is None:
                continue  # Skip invalid images
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = sp(gray, face)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                descriptors.append(np.array(face_descriptor))
                labels.append(label)

    return labels, descriptors

def compute_face_descriptor(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        shape = sp(gray, faces[0])
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        return np.array(face_descriptor)
    return None

def match_face(face_descriptor, labels, descriptors):
    distances = [distance.euclidean(face_descriptor, descriptor) for descriptor in descriptors]
    min_distance = min(distances)
    if min_distance < 1.0:  # Threshold, can be adjusted
        #falsos positivos aumente
        #falsos negativos diminua
        index = distances.index(min_distance)
        return labels[index]
    return "Unknown"

class App:
    def __init__(self, master):
        self.master = master
        self.frame_counter = 0
        self.face_labels = {}
        self.labels, self.descriptors = prepare_training_data("images")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.camera_label = Label(self.master)
        self.camera_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            self.frame_counter += 1
            if self.frame_counter % 15 == 0:  # Capture every 15 frames, adjust as needed
                new_labels = {}
                
                for face in faces:
                    shape = sp(gray, face)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    name = match_face(np.array(face_descriptor), self.labels, self.descriptors)
                    new_labels[(face.left(), face.top())] = name
                
                self.face_labels = new_labels

                # Draw labels on the frame
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    label = self.face_labels.get((x, y), "Unknown")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Ensure name is defined before using it
                if 'name' in locals():
                    photo_path = f"./predict/{name, self.frame_counter}.jpg"
                    cv2.imwrite(photo_path, frame)
                
            else:
                # Update face labels with "Unknown" for faces not in new_labels
                for face in faces:
                    if (face.left(), face.top()) not in self.face_labels:
                        self.face_labels[(face.left(), face.top())] = "Unknown"

            self.master.after(30, self.update_frame)
            
def main(args=None):
    rclpy.init(args=args)
    people_recognition_node = PeopleRecognition()
    
    root = Tk()
    app = App(root)
    root.mainloop()


    rclpy.spin(people_recognition_node)

    app.cap.release()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
