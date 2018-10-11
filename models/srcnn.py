import time
import os

from scipy import ndimage

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
        
        self.init_variables()


    #
    # Initialize parameters w1,w2,w3 and b1,b2,b3
    #
    def init_variables(self):

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

    # In article, the model is configured as below:
    #
    #       inputs --> conv(9x9, 64) --> relu --> conv(1x1, 32) --> relu --> conv(5x5, 1) --> outputs
    #
    #       1st-layer kernel size: f_1 = 9
    #       1st-layer output channels: n_1 = 64
    #       2nd-layer kernel size: f_2 = 1
    #       2nd-layer output channels: n_2 = 32
    #       3rd-layer kernel size: f_3 = 5
    #
    def model(self, images, labels):
        
        # CNN
        self.layer1 = tf.nn.relu(tf.nn.conv2d(images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
        self.layer2 = tf.nn.relu(tf.nn.conv2d(self.layer1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
        layer3 = tf.nn.conv2d(self.layer2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
        
        # Prediction
        self.pred = layer3
        
        # Loss function
        self.loss = tf.reduce_mean(tf.square(labels - self.pred))
        
        # Saver for checkpoints
        self.saver = tf.train.Saver()

    #
    # Parse dataset from tfrecord file
    #
    def _parse_function(self, example_proto):
        features = {"X": tf.FixedLenFeature([self.image_size*self.image_size*self.channels], tf.float32),
                    "Y": tf.FixedLenFeature([self.label_size*self.label_size*self.channels], tf.float32)
                   }
        parsed_features = tf.parse_single_example(example_proto, features)
        data = tf.reshape(parsed_features["X"], [self.image_size, self.image_size, self.channels])
        labels = tf.reshape(parsed_features["Y"], [self.label_size, self.label_size, self.channels])
        return data, labels

    #
    # Train model
    #
    def train(self, training_files, verbose=False):
        
        loss_hist = []
        
        # Load training data
        dataset = tf.data.TFRecordDataset(training_files, compression_type="GZIP")
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        iterator = dataset.make_initializable_iterator()
        
        # pass training data to model
        batch_images, batch_labels = iterator.get_next()
        self.model(batch_images, batch_labels)

        # Set learning rate decay
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 1, 0.9997)
        
        # Use Adam gradient descent
        optimizer = tf.train.AdamOptimizer(rate).minimize(self.loss, global_step=step)
        
        # Initialize tensorflow session
        tf.global_variables_initializer().run()
        
        # Start Training
        counter = 0
        start_time = time.time()

        for epoch in range(self.num_epoches):

            # Reset dataset iterator to begining
            self.sess.run(iterator.initializer)
            
            while True:
                try:
                    _, err = self.sess.run([optimizer, self.loss])
                except tf.errors.OutOfRangeError:
                    # when all training data consumed, OutOfRangeError will be thrown
                    break
                
                counter += 1

                if counter % 200 == 0:
                    loss_hist.append(err)

                if verbose and counter % 2000 == 0:
                    print("Epoch: %2d, step: %2d, time: %.2f, loss: %.8f" % ((epoch+1), counter, (time.time()-start_time), (err*10000)))
                    start_time = time.time()

                # save checkpoint
                if counter % 5000 == 0:
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, "srcnn.model"), global_step=counter)

        return loss_hist

                    
    #
    # Use current session to predict
    #
    def predict(self, validation_files):
        
        # Load validation data
        dataset = tf.data.TFRecordDataset(validation_files, compression_type="GZIP")
        dataset = dataset.map(map_func=self._parse_function, num_parallel_calls=4).batch(64)
        iterator = dataset.make_initializable_iterator()
        images, labels = iterator.get_next()
        self.model(images, labels)
        
        self.sess.run(iterator.initializer)
        pred = None
        loss = []
        while True:
            try:
                batch_pred, err = self.sess.run([self.pred, self.loss])
                
                # Concatenate each batch result to one array
                if pred is None:
                    pred = np.array(batch_pred)
                else:
                    pred = np.concatenate((pred, np.array(batch_pred)), axis=0)

                loss.append(err)

            except tf.errors.OutOfRangeError:
                break

        return pred, np.mean(np.array(loss))

        
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

        
    
    

        
        