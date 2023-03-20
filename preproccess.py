import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

# Set the directory containing the image data
DATADIR = "/Users/bendrummond/Documents/GitHub/seefood/data/seefood/train"

# Define the image size
IMAGE_SIZE = 60

# Define the categories of images we want to look at
CATEGORIES = ["hot_dog", "not_hot_dog"]

training_data = []

def create_training_data():
    for category in CATEGORIES:
        # Get the full path of the category directory
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        # Loop through each image file in the category directory
        for img in os.listdir(path):
            try:
                # Read the image file as a grayscale image array
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array, class_num])

            except Exception as e:
                pass

# Create the training data
create_training_data()

# Shuffle the training data
random.shuffle(training_data)

# Separate the features and labels into two arrays
X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

# Reshape the features array to have four dimensions
X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

# Save the features and labels arrays to files using pickle
with open("X.pickle", "wb") as f:
    pickle.dump(X, f)

with open("Y.pickle", "wb") as f:
    pickle.dump(Y, f)
