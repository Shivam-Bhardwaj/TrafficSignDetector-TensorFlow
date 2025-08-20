#!/usr/bin/env python3
"""
Model handling utilities for the Traffic Sign Classifier project.
This module provides functions for building, training, and evaluating the model.
Compatible with TensorFlow 2.x (no contrib).
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
import numpy as np


def build_lenet(input_shape=(32, 32, 1), num_classes=43, dropout=0.5):
    """
    Build a LeNet-style CNN using Keras Functional API.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(6, (5, 5), activation='relu', padding='valid')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (5, 5), activation='relu', padding='valid')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(84, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)


class ModelTrainer:
    def __init__(self, learning_rate=0.001, epochs=10, batch_size=128, dropout=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def train(self, X_train, y_train, X_valid, y_valid):
        model = build_lenet(dropout=self.dropout)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2
        )
        return model, history


class ModelInference:
    def __init__(self, model_path="lenet.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, images):
        return self.model.predict(images)
