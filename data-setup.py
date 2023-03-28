from typing import List

import numpy as np
import pandas as pd
import os

# Constants
TRADING_DAYS = 252
TRADE_DAYS = 160
DATA_DAYS = 90
STOP_LOSS = 0.9
STOP_WIN = 1.2

# Reading data from folder stock_data
stockPrice = np.array([])
stockNames: List[str] = list()
for file in os.listdir("stock_data/"):
    if file.endswith(".csv"):
        stockNames.append(os.path.join("stock_data/", file))
#print(stockNames)

tempName = "stock_data/Trade_history_A2M_01Feb2021.csv"
trainingData = list()
trainingLabels = list()
for name in stockNames:
    stock = np.array(pd.read_csv(name, sep=','))
    # stock = np.array(pd.read_csv(tempName, sep=','))
    close = 0
    if len(stock) >= TRADING_DAYS:
        # Limiting Trading days to a constant
        stock = stock[0:TRADING_DAYS]
        # Defining first close to base % change
        close = stock[TRADING_DAYS - 1][4]
        # Removing starting day
        stock = stock[0:TRADING_DAYS - 1]
        # Removing date, $ & % change, and volume
        stock = stock[0:TRADING_DAYS - 1, 1:5]

        # Reversing order of stock to oldest price to newest
        stock = np.flip(stock, 0)

        # Changing each price to % change of previous close
        changePercent = list()
        for day in stock:
            dayPercent = list()
            for price in day:
                percentChange = (price / close) - 1
                dayPercent.append(percentChange)
            changePercent.append(dayPercent)
            close = day[3]

        # Carrying out trades with constant variable spacings
        stockTrainingData = list()
        stockTrainingLabels = list()
        for x in range((TRADING_DAYS - 1) - TRADE_DAYS - DATA_DAYS):
            biPrice = stock[x + DATA_DAYS, 0]
            percentProfit = 0
            # Carrying out trade with stop losses, wins and expiry
            for y in range(TRADE_DAYS):
                percentProfit = stock[x + DATA_DAYS + y, 0] / biPrice
                if percentProfit <= STOP_LOSS or percentProfit >= STOP_WIN:
                    break
            priceList = list()
            for prices in changePercent[x: x + DATA_DAYS]:
                for price in prices:
                    priceList.append(price)
            stockTrainingData.append(priceList)
            percentProfit = ((1/(STOP_WIN - STOP_LOSS)) * percentProfit) + (1 - (STOP_WIN/(STOP_WIN-STOP_LOSS)))
            stockTrainingLabels.append(percentProfit)
        for data in stockTrainingData:
            trainingData.append(data)
        for label in stockTrainingLabels:
            trainingLabels.append(label)
        print(name)
print(trainingData)
print(trainingLabels)
np.savetxt("trainingData.csv", trainingData, delimiter=',')
np.savetxt("trainingLabels.csv", trainingLabels, delimiter=',')
'''
# putting all values in one array
for i in range(len(stockNames)):
    fileName = stockNames[i]
    stock = np.array(pd.read_csv(fileName, sep=',' ) )
    if len(stock) >= 254:
        stocksNo += 1

        stockData = stock[0:254, 1:5]
        for j in range(len(stockData) - 1):
            for k in range(len(stockData[j])):
                stockData[j][k] = stockData[j][k] - stockData[j + 1][3]

        change = stock[-1, 5]
        stockData[-1][0] = stockData[-1][0] - stockData[-1][3] + change
        stockData[-1][1] = stockData[-1][1] - stockData[-1][3] + change
        stockData[-1][2] = stockData[-1][2] - stockData[-1][3] + change
        stockData[-1][3] = change

        stockData = stockData / abs(np.amax(stockData))

        stockData = np.append(stockData, stock[0:254, 7:8], axis=1)
        stockData[0:254, 4:5] = stockData[0:254, 4:5] / abs(np.amax(stockData[0:254, 4:5]))
        stockData = np.reshape(stockData, (254*5))
        stockPrice = np.append( stockPrice, stockData)
print(stockPrice, stocksNo)


# split stock data into 127 dataset (6 months)
train_price = list()
stockPrice = np.reshape(stockPrice, (stocksNo * 2, 127 * 5))
for i in range(len(stockPrice)):
    if len(train_price) == 0:
        train_price = (stockPrice[i, 5:635])
    else:
        train_price = np.append(train_price, (stockPrice[i, 5:635]))

train_label_ohlc = (stockPrice[0:stocksNo * 2, 0:4])
print(train_label_ohlc)
train_labels = list()
for i in range(len(train_label_ohlc)):
    train_labels.append(train_label_ohlc[i][3] - train_label_ohlc[i][0])

for i in range(len(train_labels)):
    if (train_labels[i]) >= 0:
        train_labels[i] = 1
    else:
        train_labels[i] = 0

train_price = np.reshape(train_price, (stocksNo * 2, 126 * 5))
train_labels = np.array(train_labels, dtype=np.float)
train_price = np.array(train_price, dtype=np.float)
print("Train:",train_labels, train_price, "training data length = ",len(train_labels))
np.savetxt('train_price_OHLCV.csv', train_price, delimiter=',')
np.savetxt('train_labels_OHLCV.csv', train_labels, delimiter=',')
'''
