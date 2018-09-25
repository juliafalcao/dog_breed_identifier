import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


""" dataset import and analysis """
dataset_path = "data/kaggle_dataset/"
train_path = dataset_path + "train/"
test_path = dataset_path + "test/"

labels = pd.read_csv(dataset_path + "labels.csv")
breeds = np.array(labels["breed"].unique())

train_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_gen = ImageDataGenerator(rescale = 1./255)
train_images = train_gen.flow_from_directory(train_path, target_size = (64, 64), class_mode = "binary")
test_images = test_gen.flow_from_directory(test_path, target_size = (64, 64), class_mode = "binary")


""" build model """

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = "relu")) # 64x64 RGB (3) images
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = tf.nn.relu))
cnn.add(Dense(units = 1, activation = tf.nn.softmax)) # output

cnn.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

