import numpy as np
import pandas as pd
from tensorflow import keras
import os

# initiating variables
stockPrice = np.array([])
stockNames = list()
stocksNo = 0
for file in os.listdir("stock_data/"):
    if file.endswith(".csv"):
        stockNames.append(os.path.join("stock_data/", file))
print(stockNames)

# putting all values in one array
for i in range(len(stockNames)):
    fileName = stockNames[i]
    stock = np.array( pd.read_csv( fileName, sep=',' ) )
    if len(stock) >= 254:
        stocksNo += 1
        stock = stock[0:254, 6:7]
        stockPrice = np.append( stockPrice, stock)
print(stockPrice)

'''
# 254 point data set
train_price = list()
stockPrice = np.reshape(stockPrice, (stocksNo, 254))
for i in range(len(stockPrice)):
    if len(train_price) == 0:
        train_price = (stockPrice[i, 0:253]) / (abs(np.amax(stockPrice[i])))
    else:
        train_price = np.append(train_price, (stockPrice[i, 0:253]) / (abs(np.amax(stockPrice[i]))))
train_labels = (stockPrice[0:stocksNo, 253:254])
for i in range(len(train_labels)):
    if train_labels[i] >= 0:
        train_labels[i] = 1
    else:
        train_labels[i] = 0
train_labels = np.reshape(train_labels, len(train_labels))
train_price = np.reshape(train_price, (stocksNo, 253))
'''

# split stock data into 127 dataset (6 months)
train_price = list()
stockPrice = np.reshape(stockPrice, (stocksNo * 2, 127))
for i in range(len(stockPrice)):
    if len(train_price) == 0:
        train_price = (stockPrice[i, 0:126]) / (abs(np.amax(stockPrice[i])))
    else:
        train_price = np.append(train_price, (stockPrice[i, 0:126]) / (abs(np.amax(stockPrice[i]))))
train_labels = (stockPrice[0:stocksNo * 2, 126:127])
for i in range(len(train_labels)):
    if train_labels[i] >= 0:
        train_labels[i] = 1
    else:
        train_labels[i] = 0
train_labels = np.reshape(train_labels, len(train_labels))
train_price = np.reshape(train_price, (stocksNo * 2, 126))

train_labels = np.array(train_labels, dtype=np.float)
train_price = np.array(train_price, dtype=np.float)
print("Train:",train_labels, train_price, "training data length = ",len(train_labels))
input = len(train_price)
np.savetxt('train_price.csv', train_price, delimiter=',')
np.savetxt('train_labels.csv', train_labels, delimiter=',')


'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(253,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_price, train_labels, epochs=15)
'''
