#!/usr/bin/env python3
"""
Model handling utilities for the Traffic Sign Classifier project.
This module provides functions for building, training, and evaluating the model.
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from src.data_handler import preprocess_images
from secure_model import SecureModelHandler
import numpy as np

def LeNet(x, dropout_prob):
    """
    LeNet architecture for traffic sign classification.
    """
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_w) + fc1_b
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout_prob)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_w) + fc3_b

    return logits

class ModelTrainer:
    def __init__(self, learning_rate=0.001, epochs=10, batch_size=128, dropout=0.75):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def train(self, X_train, y_train, X_valid, y_valid):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, 43)
        keep_prob = tf.placeholder(tf.float32)

        logits = LeNet(x, keep_prob)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(self.epochs):
                X_train, y_train = self.shuffle(X_train, y_train)
                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: self.dropout})

                validation_accuracy = self.evaluate(X_valid, y_valid, sess, x, y, keep_prob, accuracy_operation)
                print(f"EPOCH {i+1} ...")
                print(f"Validation Accuracy = {validation_accuracy:.3f}")
                print()

            saver.save(sess, './lenet')
            print("Model saved")

    def evaluate(self, X_data, y_data, sess, x, y, keep_prob, accuracy_operation):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = X_data[offset:offset+self.batch_size], y_data[offset:offset+self.batch_size]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def shuffle(self, X, y):
        from sklearn.utils import shuffle
        return shuffle(X, y)

class ModelInference:
    def __init__(self, model_dir="."):
        self.handler = SecureModelHandler()
        self.handler.safe_restore_model(model_dir)

    def predict(self, images):
        processed_images = preprocess_images(np.array(images))
        
        x = self.handler.session.graph.get_tensor_by_name('x:0')
        keep_prob = self.handler.session.graph.get_tensor_by_name('keep_prob:0')
        logits = self.handler.session.graph.get_tensor_by_name('logits:0')
        
        softmax = tf.nn.softmax(logits)
        top_k = tf.nn.top_k(softmax, k=5)

        with self.handler.session.as_default():
            return self.handler.session.run(top_k, feed_dict={x: processed_images, keep_prob: 1.0})
