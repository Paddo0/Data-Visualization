import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import csv

train_labels = pd.read_csv('trainingLabels.csv', sep=',', header=None)
train_price = pd.read_csv('trainingData.csv', sep=',', header=None)
train_labels = train_labels.values.tolist()
train_price = train_price.values.tolist()
# print(train_labels, train_price)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(360,1)),
    keras.layers.Dense(360, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_price, train_labels, epochs=100)
