from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, LeakyReLU, Flatten, Dense, Lambda, UpSampling2D
from tensorflow.python.keras.engine.network import Network
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg19 import VGG19
import numpy as np
import tensorflow as tf

from models.subpixel import SubpixelConv2D

class SRGAN:
    
    def __init__(self, lr_height=96, lr_width=96, channels=3, upscaling_factor=4, gen_lr=1e-4, dis_lr=1e-4, gan_lr=1e-4):
        
        self.lr_height = lr_height
        self.lr_width = lr_width
        
        self.r = upscaling_factor
        self.hr_height = int(self.lr_height * self.r)
        self.hr_width = int(self.lr_width * self.r)
        
        self.channels = channels
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        
        # Optimizers
        optimizer_vgg = Adam(1e-4)
        optimizer_generator = Adam(gen_lr)
        #optimizer_discriminator = Adam(dis_lr)
        #optimizer_gan = Adam(gan_lr)
        
        # Build models
        #with tf.device('/gpu:1'):
        #    self.vgg = self.model_vgg(optimizer_vgg)

        self.generator = self.model_g(optimizer_generator)
        #self.discriminator, self.frozen_d = self.model_d(optimizer_discriminator)
        #self.srgan = self.model_gan(optimizer_gan)

    def model_g(self, optimizer, residual_blocks=16):
        
        # Initializators
        k_init = RandomNormal(stddev=0.02)
        g_init = RandomNormal(mean=1., stddev=0.02)
        
        n_input = Input(shape=self.lr_shape)
        n = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu', use_bias=False, kernel_initializer=k_init, name='n64s1/c')(n_input)

        # for skip connection
        temp = n

        # B residual blocks
        for i in range(residual_blocks):

            nn = Conv2D(64, (3, 3), (1, 1), 'same', name='n64s1/c1/%s' % i)(n)
            nn = BatchNormalization(name='n64s1/b1/%s' % i)(nn)
            nn = Activation('relu')(nn)

            nn = Conv2D(64, (3, 3), (1, 1), 'same', name='n64s1/c2/%s' % i)(nn)
            nn = BatchNormalization(name='n64s1/b2/%s' % i)(nn)
            nn = Activation('relu')(nn)

            n = Add(name='b_residual_add/%s' % i)([n, nn])

        n = Conv2D(64, (3, 3), (1, 1), 'same', name='n64s1/c/m')(n)
        n = BatchNormalization(name='n64s1/c/bn')(n)
        n = Add(name='add3')([n, temp])

        #n = Conv2D(256, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n256s1/1')(n)
        #n = SubpixelConv2D(scale=2, activation='relu', name='pixelshufflerx2/1')(n)

        #n = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n256s1/2')(n)
        #n = SubpixelConv2D(scale=2, activation='relu', name='pixelshufflerx2/2')(n)
        
        for i in range(int(np.log(self.r) / np.log(2))):
            n = UpSampling2D(size=2)(n)
            n = Conv2D(256, (3,3), (1,1), padding='same', activation='relu')(n)

        n = Conv2D(3, (3, 3), (1, 1), 'same', activation='tanh', kernel_initializer=k_init, name='out')(n)
        
        # Create model and compile
        model = Model(inputs=n_input, outputs=n)
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def model_d(self, optimizer, filters=16):
        
        # Initializators
        k_init = RandomNormal(stddev=0.02)
        g_init = RandomNormal(mean=1., stddev=0.02)

        n_input = Input(shape=self.hr_shape)

        n = Conv2D(filters, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h0/c')(n_input)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*2, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h1/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h1/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*4, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h2/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h2/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*8, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h3/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h3/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*16, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h4/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h4/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*32, (4, 4), (2, 2), 'same', use_bias=False, kernel_initializer=k_init, name='h5/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h5/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*16, (1, 1), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='h6/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h6/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*8, (1, 1), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='h7/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='h7/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        temp = n

        n = Conv2D(filters*2, (1, 1), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='res/c')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='res/bn')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*2, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='res/c2')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='res/bn2')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Conv2D(filters*8, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='res/c3')(n)
        n = BatchNormalization(gamma_initializer=g_init, name='res/bn3')(n)
        n = LeakyReLU(alpha=0.2)(n)

        n = Add(name='res/add')([temp, n])
        n = LeakyReLU(alpha=0.2)(n)

        n = Flatten()(n)
        n = Dense(1, activation='sigmoid', use_bias=False, kernel_initializer=k_init, name='ho/dense')(n)

        # Create model and compile
        model = Model(inputs=n_input, outputs=n)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Fixed model for gan
        frozen_model = Network(inputs=n_input, outputs=n)
        frozen_model.trainable = False
        
        return model, frozen_model

    # VGG19 network
    def model_vgg(self, optimizer):
        
        n_input = Input(shape=self.hr_shape)
        n = Lambda(lambda x: tf.image.resize_images(x, size=[224, 224], method=0, align_corners=False))(n_input)
        
        # Get the vgg network, extract features from last conv layer
        vgg = VGG19(pooling='max')
        vgg.outputs = [vgg.layers[16].output]
        
        #Create model and compile
        model = Model(inputs=n_input, outputs=vgg(n))
        model.trainable = False
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        
        return model
    
    
    def model_gan(self, optimizer):
        
        # Input LR images
        img_lr = Input(self.lr_shape)
        
        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        
        # Extract features using VGG
        generated_features = self.vgg(generated_hr)
        
        # Run discriminator on generated high resolution image
        generated_check = self.frozen_d(generated_hr)
        
        # Create model and compile
        model = Model(inputs=img_lr, outputs=[generated_check, generated_features, generated_hr])
        model.compile(
            loss=['binary_crossentropy', 'mse', 'mse'],
            loss_weights=[1e-3, 5e-4, 1],
            optimizer=optimizer
        )
        
        return model
        
        
    def save_weights(self, path):
        # Save generator and discriminator network
        self.generator.save_weights(path + "_generator.h5")
        self.discriminator.save_weights(path + "_discriminator.h5")
        
    def load_weights(self, generator_weights=None, discriminator_weights=None):
        if generator_weights:
            self.generator.load_weights(generator_weights)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights)
        