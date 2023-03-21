import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time
import pickle



# Load the training data from pickle files
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

# Convert Y to numpy array for compatibility with tensorflow
Y = np.array(Y)

# Convert X to numpy array and normalize pixel values
X = np.array(X) / 255.0

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]


# dense_layers = [0]
# layer_sizes = [32]
# conv_layers = [1]
# Print the shape of the data arrays
print("X shape:", X.shape)
print("Y shape:", Y.shape)

for dense_layer in dense_layers:
  for layer_size in layer_sizes:
    for conv_layer in conv_layers:
      NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
      tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
      print(NAME)
      model = Sequential()
      model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
      model.add(Activation("relu"))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      
      for l in range(conv_layer-1):
        model.add(Conv2D(layer_size, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        
      model.add(Flatten()) 
      for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        
    
      model.add(Dense(1))
      model.add(Activation('sigmoid'))

      # Compile the model with binary crossentropy loss and Adam optimizer
      model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

      # Fit the model to the training data with a batch size of 2, validation split of 0.1, and 10 epochs
      model.fit(X, Y, batch_size=10, validation_split=0.1, epochs=10, callbacks=[tensorboard])

import time




