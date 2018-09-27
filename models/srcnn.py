import time
import os

from scipy import ndimage
from six.moves import cPickle as pickle

import numpy as np
import tensorflow as tf

class SRCNN:
    
    def __init__(self,
                 sess,
                 image_size=33,
                 label_size=21,
                 scale = 3,
                 stride = 14,
                 batch_size=128,
                 num_epoches=50000,
                 channels=1,
                 learning_rate=1e-4
                 checkpoint_dir=None,
                 sample_dir=None):
        
        self.sess = sess
        self.is_grayscale = (channels == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.scale = scale
        self.stride = 14
        self.batch_size = batch_size

        self.channels = channels

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir


    # As per article, the model is configured as below:
    #
    #       inputs --> conv(9x9, 64) --> relu --> conv(1x1, 32) --> relu --> conv(5x5, 1) --> outputs
    #
    #       1st-layer kernel size: f_1 = 9
    #       1st-layer output channels: n_1 = 64
    #       2nd-layer kernel size: f_2 = 1
    #       2nd-layer output channels: n_2 = 32
    #       3rd-layer kernel size: f_3 = 5
    #
    def build_model(self):

        # Initialize inputs and parameters
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.channels], name='images')
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.channels], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        
        # CNN
        layer1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        layer2 = tf.nn.relu(tf.nn.conv2d(layer1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        layer3 = tf.nn.conv2d(layer2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
        
        self.predict = layer3
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(self.labels - self.predict))
        
        # Saver for checkpoints
        self.savers = tf.train.Saver()


    #
    # Train model
    #
    def train(self, data, label):
        

        # Use Adam gradient descent
        optimizer = tf.train.AdmamOptimizer(lr).minimize(self.loss)
        
        tf.initialize_all_variables().run()
        
        # Start Training
        counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            
            num_batches = data.shape[0] // self.batch_size
            
            for step in range(num_batches):
                
                offset = step * self.batch_size
                batch_images = data[offset:(offset+self.batch_size), :, : ,:]
                batch_labels = label[offset:(offset+self.batch_size), :]
                
                counter += 1
                _, err = self.sess.run([optimizer, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                
                if counter % 1000 == 0:
                    print("Epoch: %2d, step: %2d, time: %4.4f, loss: %.8f" % ((epoch+1), counter, time.time-start_time, err))

                if counter % 5000 == 0:
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "SRCNN.model"), global_step=counter)


                    
    #
    # Use current session to predict
    #
    def predict(self):
        
        result = self.predict.eval({self.images: None, self.labels: None})

        
    #
    # Load check point from saved ckpt file
    #
    # Input:
    #        checkpoint: number of checkpoint
    #
    def load_checkpoint(self, checkpoint):
        
        #ckpt_dir = os.path.join(
        


    def load_data(self, is_train=True):
        
        if is_train:
            path = os.path.join(self.checkpoint_dir, 'train.pickle')
        else:
            path = os.path.join(self.checkpoint_dir, 'validate.pickle')

        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            input_ = dataset['data']
            label_ = dataset['labels']

        return input_, label_
        
    
    def preprocess(self, dataset_dir, is_train=True):
        
        sub_input_sequence = []
        sub_label_sequence = []
        padding = abs(self.image_size - self.label_size)
        pixel_depth = 255.0

        for filename in os.listdir(dataset_dir):
            
            path = os.path.join(dataset_dir, filename)
            image = (scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float) - pixel_depth/2)/pixel_depth
            #if is_grayscale:
            #else:
            #    image = (scipy.misc.imread(path, mode='YCbCr').astype(np.float) - pixel_depth/2)/pixel_depth

            label_ = self.modcrop(image, self.scale)
            if len(image.shape) == 3:
                h, w, _ = image.shape
                h = h - np.mod(h, self.scale)
                w = w - np.mod(w, self.scale)
                image = image[0:h, 0:w, :]
            else:
                h, w = image.shape
                h = h - np.mod(h, self.scale)
                w = w - np.mod(w, self.scale)
                image = image[0:h, 0:w]
            
            label_ = image
            input_ = ndimage.interpolation.zoom(label_, (1./self.scale), prefilter=False)
            input_ = ndimage.interpolation.zoom(input_, self.scale, prefilter=False)
            
            for x in range(0, h-self.image_size+1, self.stride):
                for y in range(0, w-self.image_size+1, self.stride):
                    sub_input = input_[x:x+self.image_size, y:y+self.image_size] # [33 x 33]
                    sub_label = label_[x+int(padding):x+int(padding)+self.label_size, y+int(padding):y+int(padding)+self.label_size] # [21 x 21]

                    # Make channel value
                    sub_input = sub_input.reshape([self.image_size, self.image_size, 1])  
                    sub_label = sub_label.reshape([self.label_size, self.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

        arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
        arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
        
        if is_train:
            savepath = os.path.join(self.checkpoint_dir, 'train.pickle')
        else:
            savepath = os.path.join(self.checkpoint_dir, 'validate.pickle')
        
        with open(savepath, 'wb') as f:
            pickle.dump({'data': arrdata,
                         'labels': arrlabel}, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        
        return arrdata, arrlabel

        
        