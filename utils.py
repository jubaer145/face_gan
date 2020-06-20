from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
from numpy import savez_compressed
import os



def load_image(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    return pixels

def plot_faces(faces, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(faces[i])
    plt.show()

def extract_face(model, pixels, size = (80, 80)):
    faces = model.detect_faces(pixels)
    if len(faces) == 0:
        return None
    x1, y1, w, h = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h
    face_pixels = pixels[y1 : y2, x1 : x2]
    image = Image.fromarray(face_pixels)
    image = image.resize(size)
    face = np.asarray(image)
    return face

def load_faces(directory, n_faces):
    model = MTCNN()
    faces = list()
    for filename in os.listdir(directory):
        pixels = load_image(directory + filename)
        face = extract_face(model, pixels)
        if face is None:
            continue
        faces.append(face)
        print(f"{len(faces)} {face.shape}")
        if len(faces) >= n_faces:
            break
    return np.asarray(faces)
