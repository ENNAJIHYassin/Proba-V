from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add

class SRGAN():

    """ Model Architecture is inspired from the Paper:
        https://arxiv.org/pdf/1609.04802.pdf   """

    def __init__(self):
        # Input / Output shape
        self.channels = 1
        self.lr_dim = 128                 # Low resolution dimension
        self.lr_shape = (self.lr_dim, self.lr_dim , self.channels)
        self.hr_dim = 384                 # High resolution dimesion
        self.hr_shape = (self.hr_dim, self.hr_dim, self.channels)

    @staticmethod
    def res_block_gen(model, kernal_size, filters, strides):

        gen = model

        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        model = BatchNormalization()(model) #momentum = 0.5?0.8?
        model = PReLU(shared_axes=[1,2])(model)
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        model = BatchNormalization()(model) #momentum = 0.5?0.8?

        model = add([gen, model])

        return model

    @staticmethod
    def up_sampling_block(model, kernal_size, filters, strides):

        """ As per suggestion from http://distill.pub/2016/deconv-checkerboard/, we are using UpSampling2D as a
            simple Nearest Neighbour Upsampling instead of SubPixelConvolution   """

        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        model = LeakyReLU(alpha = 0.25)(model)
        model = UpSampling2D(size = 3)(model)
        model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        model = LeakyReLU(alpha = 0.3)(model)

        return model

    @staticmethod
    def discriminator_block(model, filters, kernel_size, strides):

        model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
        model = BatchNormalization()(model) #momentum = 0.5?0.8?
        model = LeakyReLU(alpha = 0.2)(model)

        return model

    def generator(self):

        gen_input = Input(shape = self.lr_shape)
        model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
        model = PReLU(shared_axes=[1,2])(model)

        gen_model = model

        # Using 16 Residual Blocks
        for _ in range(16):
            model = self.res_block_gen(model, 3, 64, 1)

        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([gen_model, model])

        # Using 1 UpSampling Block
        model = self.up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters = self.channels, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('tanh')(model)

        generator_model = Model(inputs = gen_input, outputs = model)

        return generator_model

    def discriminator(self):

        dis_input = Input(shape = self.hr_shape)

        model = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)

        """ Strided convolutions are used to reduce the image resolution each
            time the number of features is doubled."""
        model = self.discriminator_block(model, 32, 3, 2)
        model = self.discriminator_block(model, 64, 3, 1)
        model = self.discriminator_block(model, 64, 3, 2)
        model = self.discriminator_block(model, 128, 3, 1)
        model = self.discriminator_block(model, 128, 3, 2)
        model = self.discriminator_block(model, 256, 3, 1)
        model = self.discriminator_block(model, 256, 3, 2)

        model = Flatten()(model)
        model = Dense(512)(model)
        model = LeakyReLU(alpha = 0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs = dis_input, outputs = model)

        return discriminator_model

    def srgan(self, generator, discriminator):

        discriminator.trainable = False
        srgan_in = Input(shape=self.lr_shape)
        gen_out = generator(srgan_in)
        srgan_out = discriminator(gen_out)

        return Model(srgan_in, [gen_out, srgan_out])
