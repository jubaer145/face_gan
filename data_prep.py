from utils import *

dir_path = '/media/jubaer/DataBank/Fashion_GAN/img_align_celeba/'
face_collection = load_faces(dir_path, 50000)
print(f"Loaded: {face_collection.shape}")
savez_compressed('./img_align_celeba.npz', face_collection)