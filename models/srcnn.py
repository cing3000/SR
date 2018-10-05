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
                 batch_size=128,
                 num_epoches=50000,
                 channels=1,
                 stddev=1e-3,
                 learning_rate=1e-4,
                 checkpoint_dir=None,
                 sample_dir=None):
        
        self.sess = sess
        self.is_grayscale = (channels == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.num_epoches = num_epoches
        self.channels = channels
        self.learning_rate = learning_rate
        self.stddev = stddev

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        
        self.build_model()


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
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.channels], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.channels, 64], stddev=self.stddev), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=self.stddev), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.channels], stddev=self.stddev), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([self.channels]), name='b3')
        }
        
        # CNN
        self.layer1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        self.layer2 = tf.nn.relu(tf.nn.conv2d(self.layer1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        layer3 = tf.nn.conv2d(self.layer2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
        
        self.pred = layer3
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        
        # Saver for checkpoints
        self.saver = tf.train.Saver()


    #
    # Train model
    #
    def train(self, image, label, verbose=False):
        
        loss_hist = []

        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 1, 0.9997)
        
        # Use Adam gradient descent
        optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)
        #optimizer = tf.train.GradientDescentOptimizer(rate).minimize(self.loss, global_step=step)
        
        tf.global_variables_initializer().run()
        
        # Start Training
        counter = 0
        start_time = time.time()
        
        for epoch in range(self.num_epoches):
            
            num_batches = image.shape[0] // self.batch_size
            
            for batch in range(num_batches):
                
                offset = batch * self.batch_size
                batch_images = image[offset:(offset+self.batch_size), :, : ,:]
                batch_labels = label[offset:(offset+self.batch_size), :]
                
                counter += 1
                _, err = self.sess.run([optimizer, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                
                if counter % 200 == 0:
                    loss_hist.append(err)

                if verbose and counter % 2000 == 0:
                    print("Epoch: %2d, step: %2d, loss: %.8f" % ((epoch+1), counter, (err*10000)))

                if counter % 5000 == 0:
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "srcnn.model"), global_step=counter)

        return loss_hist

                    
    #
    # Use current session to predict
    #
    def predict(self, image):
        pred, weights = self.sess.run([self.pred, self.weights], feed_dict={self.images: image})
        return pred, weights

        
    #
    # Load check point from saved ckpt file
    #
    # Input:
    #        checkpoint: number of checkpoint
    #
    def load_checkpoint(self, checkpoint):

        #ckpt_dir = os.path.join(checkpoint_dir, 'srcnn.model-' + checkpoint)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            print(os.path.join(self.checkpoint_dir, ckpt_name))

            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            return True
        else:
            return False

        
    
    

        
        