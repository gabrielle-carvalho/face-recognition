import cv2
import os
import interface2b
import dlib
import numpy as np
from scipy.spatial import distance
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Cria a pasta 'imagens' se ela não existir
if not os.path.exists('images'):
    os.makedirs('images')

# Inicializa a captura de vídeo (0 geralmente é a webcam padrão)
cap = cv2.VideoCapture(0)

# Verifica se a captura foi aberta com sucesso
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

print("Pressione 's' para tirar uma foto ou 'q' para sair")

img_count = 0

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

while True:
    # Captura frame a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar imagem")
        break
    
    # Exibe a imagem capturada
    cv2.imshow('Webcam', frame)

    # Espera uma tecla ser pressionada
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Nome do arquivo com contador
        img_name = f"images/foto_{img_count}.png"
        # Salva a imagem
        cv2.imwrite(img_name, frame)
        print(f"Imagem salva como {img_name}")
        img_count += 1

    elif key == ord('q'):
        # Sai do loop
        print("Saindo...")
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()


#recebe o nome da pessoa -> cria a pasta com o nome da pessoa -> tira uma quantidade específica de fotos da pessoa -> guarda essas fotos na pasta com o nome dela

