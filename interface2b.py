import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
key = cv2. waitKey(1)
img_count = 0

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


def compute_face_descriptor(frame):
    faces = detector(frame)
    if len(faces) > 0:
        shape = sp(frame, faces[0])
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        return np.array(face_descriptor)
    return None


def match_face(face_descriptor, labels, descriptors):
    distances = [distance.euclidean(face_descriptor, descriptor) for descriptor in descriptors]
    min_distance = min(distances)
    if min_distance < 0.9:  # Threshold, can be adjusted
        index = distances.index(min_distance)
        return labels[index]
    return "Unknown"



class App:
    def __init__(self, master):
        self.master = master
        self.master.title('Face Recognition Software')
        self.master.geometry('1000x1000')
        self.frame_counter = 0
        self.recognized_faces = {}
        self.labels, self.descriptors = prepare_training_data("images")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.camera_label = Label(self.master)
        self.camera_label.pack()

        self.name_label = Label(self.master, text="Enter Name:")
        self.name_label.pack()

        self.name_entry = Entry(self.master)
        self.name_entry.pack()

        self.capture_button = Button(self.master, text="Capture", command=self.on_capture_click)
        self.capture_button.pack(side=LEFT)

        self.retrain_button = Button(self.master, text="Retrain", command=self.on_retrain_click)
        self.retrain_button.pack(side=LEFT)

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            self.frame_counter += 1
            if self.frame_counter % 1 == 0:
                new_labels = {}
                for face in faces:
                    shape = sp(gray, face)
                    face_descriptor = facerec.compute_face_descriptor(frame, shape)
                    name = match_face(np.array(face_descriptor), self.labels, self.descriptors)
                    new_labels[(face.left(), face.top())] = name
                    img_count =+ 1
                    
                    
                self.face_labels = new_labels

                

            else:
                for face in faces:
                    if (face.left(), face.top()) not in self.face_labels:
                        self.face_labels[(face.left(), face.top())] = "Unknown"

            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = self.face_labels.get((x, y), "Unknown")
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            frame = cv2.resize(frame, (1000, int(1000 * frame.shape[0] / frame.shape[1])))
            image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.camera_label.configure(image=image)
            self.camera_label.image = image
            self.master.after(30, self.update_frame)
            photo = f"./predict/{name, img_count}.jpg"
            cv2.imwrite(photo, frame)
            # foi feito: coloquei para cada label que ele reconhece, uma foto de predict e armazenar na pasta predict
            

        
            
        
    def on_capture_click(self):
        name = self.name_entry.get()
        ret, frame = self.cap.read()

        if ret:
            path = f"./images/{name}"
            if not os.path.exists(path):
                os.mkdir(path)
            picnumber = len(os.listdir(path))
            cv2.imwrite(f'{path}/{picnumber}.png', frame)
            messagebox.showinfo("Info", "Image Captured Successfully!")

    def on_retrain_click(self):
        self.labels, self.descriptors = prepare_training_data("images")
        messagebox.showinfo("Info", "Model Retrained with Updated Images!")


def main():
    root = Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
