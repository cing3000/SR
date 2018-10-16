from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.initializers import RandomNormal

from subpixel import SubpixelConv2d

class SRGAN:
    
    def __init__(self):
        pass
    
    
    def model_g(trainable=False, reuse=False):
        
        # Initializators
        k_init = RandomNormal(stddev=0.02)
        g_init = RandomNormal(mean=1. stddev=0.02)

        with tf.variable_scope("srgan_g", reuse=reuse) as vs:
            
            model = Sequential()

            n = Input(shape=(96,96,3), name='in')
            n = Conv2D(64, (3, 3), (1, 1), 'same', activation='relu', use_bias=False, kernel_initializer=k_init, name='n64s1/c')(n)

            # for skip connection
            temp = n

            # B residual blocks
            for i in range(16):

                nn = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n64s1/c1/%s' % i)(n)
                nn = BatchNormalization(gamma_initializer=g_init, trainable=trainable, name='n64s1/b1/%s' % i)(nn)
                nn = Activation('relu')(nn)

                nn = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n64s1/c2/%s' % i)(nn)
                nn = BatchNormalization(gamma_initializer=g_init, trainable=trainable, name='n64s1/b2/%s' % i)(nn)
                nn = Activation('relu')(nn)

                nn = Add(name='b_residual_add/%s' % i)([n, nn])

                n = nn

            n = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n64s1/c/m')(n)
            n = BatchNormalization(gamma_initializer=g_init, trainable=trainable, name='n64s1/c/m')(n)
            n = Add(name='add3')([n, temp])

            n = Conv2D(256, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n256s1/1')(n)
            n = SubpixelConv2d(scale=2, activation='relu', name='pixelshufflerx2/1')(n)

            n = Conv2D(64, (3, 3), (1, 1), 'same', use_bias=False, kernel_initializer=k_init, name='n256s1/2')(n)
            n = SubpixelConv2d(scale=2, activation='rule', name='pixelshufflerx2/2')(n)

            n = Conv2D(3, (1, 1), (1, 1), 'same', activation='tanh', kernel_initializer=k_init, name='out')(n)
            
            return n

    def model_d(trainable=False, reuse=False):
        
        # Initializators
        k_init = RandomNormal(stddev=0.02)
        g_init = RandomNormal(mean=1. stddev=0.02)
        
        