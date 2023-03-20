import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

# Load the training data from pickle files
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

# Convert Y to numpy array for compatibility with tensorflow
Y = np.array(Y)

# Convert X to numpy array and normalize pixel values
X = np.array(X) / 255.0

# Print the shape of the data arrays
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Define the neural network model
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1, activation="sigmoid"))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model to the training data with a batch size of 2, validation split of 0.1, and 10 epochs
model.fit(X, Y, batch_size=2, validation_split=0.1, epochs=10)
