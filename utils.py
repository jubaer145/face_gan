from PIL import Image
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
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

def load_real_samples():
    data  = np.load('img_align_celeba.npz')
    X = data['arr_0']
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return X

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y  = np.ones((n_samples, 1))
    return X, y

#generate vector of points sampled from latent space
def generate_latent_points(latent_dim, n_samples):
    x_in = np.random.randn(latent_dim * n_samples)
    x_in = x_in.reshape(n_samples, latent_dim)
    return x_in

#use the generator to generate fake samples with class label
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y

#create and save a plot of generated images
def save_plot(examples, epoch, n = 10):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis('off')
        plt.imshow(examples[i])
    file_name = f"generated_plot_e{epoch + 1 : 0.3f}.png"
    save_path = "./generated_image" + file_name
    plt.savefig(save_path)
    plt.close()
    
#evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_samples = 100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose = 0)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n_samples)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose = 0)
    print(f">Accuracy real: {acc_real * 100 : .0f}%%, fake: = {acc_fake * 100 : .0f}%%")
    save_plot(x_fake, epoch)
    file_name = f"generator_model_{epoch + 1}.h5"
    save_path = "./saved_models" + file_name
    generator.save(save_path)


