import pandas as pd
import numpy as np
import neat
import pickle

with open("winnersetupOHLC.p", "rb") as f:
    network = pickle.load(f)
    network = neat.nn.FeedForwardNetwork.create(network[0], network[1])

fileName = 'stock_data\Trade_history_A2M_03Feb2021.csv'
stock = np.array(pd.read_csv(fileName, sep=','))
stockData = stock[1:127, 1:5]
for j in range(len(stockData) - 1):
    for k in range(len(stockData[j])):
        stockData[j][k] = stockData[j][k] - stockData[j + 1][3]

change = stock[-1, 5]
stockData[-1][0] = stockData[-1][0] - stockData[-1][3] + change
stockData[-1][1] = stockData[-1][1] - stockData[-1][3] + change
stockData[-1][2] = stockData[-1][2] - stockData[-1][3] + change
stockData[-1][3] = change

stockData = stockData / abs(np.amax(stockData))

stockData = np.append(stockData, stock[0:126, 7:8], axis=1)
stockData[0:126, 4:5] = stockData[0:126, 4:5] / abs(np.amax(stockData[0:126, 4:5]))
stockData = np.reshape(stockData, (126 * 5))

output = network.activate(stockData)
print(output)