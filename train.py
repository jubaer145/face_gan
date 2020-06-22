
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from model import *
from utils import *


#train the generator and the discriminator
def train(generator, discriminator, gan, dataset, latent_dim,  n_epochs = 50, batch_size = 8):
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    for i in range(n_epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss_1, _ = discriminator.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
            d_loss_2, _ = discriminator.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, batch_size)
            y_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(X_gan, y_gan)
            print(f">{i + 1}, {j + 1}, {batch_per_epoch}, d_loss_1 = {d_loss_1: 0.3f}, d_loss_2 = {d_loss_2: 0.3f}, g_loss = {g_loss: 0.3f}")
        if (i + 1) % 2 == 0:
            summarize_performance(i, generator, discriminator, dataset, latent_dim)


latent_dim = 100
discriminator = define_discriminator()
generator = define_genertaor(latent_dim)
gan = define_gan(generator, discriminator)
dataset = load_real_samples()
train(generator, discriminator,gan, dataset, latent_dim)