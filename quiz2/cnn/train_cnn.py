import time
import numpy as np
import math


import tensorflow as tf

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode
    def bias_variable(self,shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        X1 = tf.reshape(X,[-1,784])

        W = tf.Variable( tf.truncated_normal([784, hidden_size],
                            stddev=1.0 / math.sqrt(float(784))))
        b = tf.Variable(tf.zeros([hidden_size]))
        W1 = tf.Variable(tf.truncated_normal([hidden_size,10],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b1 = tf.Variable(tf.zeros([10]))
        h1 = tf.sigmoid(tf.matmul(X1, W) + b)
        y = tf.matmul(h1, W1) + b1
        #y = tf.nn.softmax(tf.matmul(h1, W1) + b1)
        
        return y

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        
        conv1 = tf.layers.conv2d(X, 20, 5, activation=tf.nn.sigmoid)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 40, 5, activation=tf.nn.sigmoid)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        X1 = tf.reshape(conv2,[-1,640])

        W = tf.Variable( tf.truncated_normal([640, hidden_size],
                            stddev=1.0 / math.sqrt(float(640))))
        b = tf.Variable(tf.zeros([hidden_size]))
        W1 = tf.Variable(tf.truncated_normal([hidden_size,10],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b1 = tf.Variable(tf.zeros([10]))
        h1 = tf.sigmoid(tf.matmul(X1, W) + b)
        y = tf.matmul(h1, W1) + b1
        #y = tf.matmul(h1, W1) + b1
        #y = tf.nn.softmax(tf.matmul(h1, W1) + b1)
        
        return y

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        
        conv1 = tf.layers.conv2d(X, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        X1 = tf.reshape(conv2,[-1,640])

        W = tf.Variable( tf.truncated_normal([640, hidden_size],
                            stddev=1.0 / math.sqrt(float(640))))
        b = self.bias_variable([hidden_size])
        W1 = tf.Variable(tf.truncated_normal([hidden_size,10],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b1 = self.bias_variable([10])
        h1 = tf.nn.relu(tf.matmul(X1, W) + b)
        y = tf.matmul(h1, W1) + b1
        #y = tf.matmul(h1, W1) + b1
        #y = tf.nn.softmax(tf.matmul(h1, W1) + b1)
        
        return y

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        
        conv1 = tf.layers.conv2d(X, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        X1 = tf.reshape(conv2,[-1,640])

        W = tf.Variable( tf.truncated_normal([640, hidden_size],
                            stddev=1.0 / math.sqrt(float(640))))
        b = self.bias_variable([hidden_size])
        W1 = tf.Variable(tf.truncated_normal([hidden_size,hidden_size],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b1 = self.bias_variable([hidden_size])
        W2 = tf.Variable(tf.truncated_normal([hidden_size,10],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b2 = self.bias_variable([10])
        h1 = tf.nn.relu(tf.matmul(X1, W) + b)
        h2 = tf.nn.relu(tf.matmul(h1, W1) + b1)
        y = tf.matmul(h2, W2) + b2
        weight1 = tf.nn.l2_loss(W1)
        weight2 = tf.nn.l2_loss(W2)
        #y = tf.matmul(h1, W1) + b1
        #y = tf.nn.softmax(tf.matmul(h1, W1) + b1)
        
        return y,weight1+weight2

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        conv1 = tf.layers.conv2d(X, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 40, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        X1 = tf.reshape(conv2,[-1,640])
        
        W = tf.Variable( tf.truncated_normal([640, hidden_size],
                            stddev=1.0 / math.sqrt(float(640))))
        b = self.bias_variable([hidden_size])
        W1 = tf.Variable(tf.truncated_normal([hidden_size,hidden_size],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b1 = self.bias_variable([hidden_size])
        W2 = tf.Variable(tf.truncated_normal([hidden_size,10],
                            stddev=1.0 / math.sqrt(float(hidden_size))))
        b2 = self.bias_variable([10])
        h1 = tf.nn.relu(tf.matmul(X1, W) + b)
        h1 = tf.nn.dropout(h1, 0.5)
        h2 = tf.nn.relu(tf.matmul(h1, W1) + b1)

        weight1 = tf.nn.l2_loss(W1)
        weight2 = tf.nn.l2_loss(W2)
        
        h_fc1_drop = tf.nn.dropout(h2, 0.5)
        y = tf.matmul(h_fc1_drop, W2) + b2
#         weight1 = tf.nn.l2_loss(W)
#         weight2 = tf.nn.l2_loss(W1)
        #y = tf.matmul(h1, W1) + b1
        #y = tf.nn.softmax(tf.matmul(h1, W1) + b1)
        
        return y,weight1+weight2

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set,outfile):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)


#         b = np.zeros((trainY.shape[0], class_num))
#         b[np.arange(trainY.shape[0]), trainY] = 1
#         trainY = b

        
        

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.int64, [None])
#             depth = 10
#             Y1 = tf.one_hot(Y, depth)
#             print Y1.shape

            is_train = tf.placeholder(tf.bool)
            

            # model 1: base line
            if self.mode == 1:

                
                features = self.model_1(X, hidden_size)
                weight = 0
                



            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)
                weight = 0

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)
                weight = 0


            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features ,weight= self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features,weight = self.model_5(X, hidden_size, is_train)
                
            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to logits after code implementation.
            logits = features
            beta = 0.0001
            
            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            # Remove NotImplementedError and assign calculated value to loss after code implementation.
            labels = tf.to_int64(Y)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
            loss = tf.reduce_mean(cross_entropy+beta*weight, name='xentropy_mean')
            #loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            #loss = tf.reduce_mean(-tf.reduce_sum(Y1 * tf.log(logits), reduction_indices=[1]))


            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------

            # Remove NotImplementedError and assign calculated value to train_op after code implementation.
            train_op  = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            #train_op  = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            correct_prediction = tf.equal(tf.argmax(logits,1),labels )
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    outfile.write(20 * '*'+ 'epoch'+str(i + 1)+ 20 * '*'+'\n')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]

                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))
                    outfile.write('the training took: %d(s)\n' % (end_time - start_time))

                    total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct ))
                    outfile.write('accuracy of the trained model %f\n' % (total_correct ))
                    print ()

                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) 





