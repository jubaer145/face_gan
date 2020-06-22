import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, LeakyReLU, Reshape, Flatten, Dropout
from keras.optimizers import Adam

#define the discriminator model
def define_discriminator(in_size = (80, 80, 3)):
    inp = Input(shape = in_size)
    conv_1 = Conv2D(128, (5, 5), padding = 'same')(inp)
    lrelu_1 = LeakyReLU(alpha = 0.2)(conv_1)
    conv_2 = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(lrelu_1)
    lrelu_2 = LeakyReLU(alpha = 0.2)(conv_2)
    conv_3 = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(lrelu_2)
    lrelu_3 = LeakyReLU(alpha = 0.2)(conv_3)
    conv_4 = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(lrelu_3)
    lrelu_4 = Conv2D(128, (5, 5), strides = (2, 2), padding = 'same')(conv_4)
    flatten = Flatten()(lrelu_4)
    drop_1 = Dropout(0.4)(flatten)
    dense_1 = Dense(1, activation = 'sigmoid')(drop_1)
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model = Model(inputs = inp, outputs = dense_1)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model

#define the generator model
def define_genertaor(latent_dim):

    # base for 5 x 5 images
    n_nodes = 128 * 5 * 5
    inp = Input(shape = (latent_dim, ))
    dens_1 = Dense(n_nodes)(inp)
    lrelu_1 = LeakyReLU(alpha = 0.2)(dens_1)
    reshape_1 = Reshape((5, 5, 128))(lrelu_1)
    conv_tr_1 = Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same')(reshape_1)
    lrelu_2 = LeakyReLU(alpha = 0.2)(conv_tr_1)
    conv_tr_2 = Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same')(lrelu_2)
    lrelu_3 = LeakyReLU(alpha = 0.2 )(conv_tr_2)
    conv_tr_3 = Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same')(lrelu_3)
    lrelu_4 = LeakyReLU(alpha = 0.2)(conv_tr_3)
    conv_tr_4 = Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same')(lrelu_4)
    lrelu_5 = LeakyReLU(alpha = 0.2)(conv_tr_4)
    conv_1 = Conv2D(3, (5, 5), activation = 'tanh', padding = 'same')(lrelu_5)
    model = Model(inputs = inp, outputs = conv_1)
    return model

#define GAN
def define_gan(generator, discriminator):
    discriminator.trainable = False
    gan = Model(inputs = generator.inputs, outputs = discriminator(generator.outputs))
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    gan.compile(loss = 'binary_crossentropy', optimizer = opt)
    return gan
