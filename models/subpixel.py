import tensorflow as tf
import keras

class SubpixelConv2D(Layer):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    
    def __init__(self, scale=2, actitvation=None, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.scale = scale
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        
        
    def call(self, inputs):
        outputs = tf.depth_to_space(inputs, self.scale)
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        dims = [input_shape[0],
                input_shape[1] * self.scale,
                input_shape[2] * self.scale,
                int(input_shape[3] / (self.scale ** 2))]
        output_shape = tuple(dims)
        return output_shape