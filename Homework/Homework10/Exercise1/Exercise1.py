# -*- coding: utf-8 -*-
"""
MNISTweights.ipynb
Automatically generated by Colab.
Original file is located at
https://colab.research.google.com/drive/1n5S1ooyt6u4-oSgZ7vdB6OtGw1h7qDVY
"""
from tensorflow.keras.datasets import mnist
import cv2
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
print(y_train)
print(y_train.shape)
print(y_train[0])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f'categorical data:\n{y_train}')
print(y_train.shape)
print(y_train[0])

cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
cnn.add(MaxPool2D(pool_size=(2, 2)))
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2, 2)))
# added layers
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
cnn.add(MaxPool2D(pool_size=(2, 2)))
#
cnn.add(Flatten())
cnn.add(Dense(units=128, activation="relu"))
cnn.add(Dense(units=10, activation="softmax"))

cnn.summary()
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = cnn.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
_, test_acc = cnn.evaluate(X_test, y_test, verbose=0)

print(f'Training accuracy: {acc}')
print(f'Validation accuracy: {val_acc}')
print(f'Test accuracy: {test_acc}')

# Evaluate accuracy on test dataset...
# Convert y_test to one-hot encoded format
print(f'y_test shape (before one-hot encoding): {y_test.shape}')
y_test = to_categorical(y_test) # convert to one-hot encoded labels (shape (None,10)
print(f'y_test shape (after one-hot encoding): {y_test.shape}')
