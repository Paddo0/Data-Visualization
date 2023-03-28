import pandas as pd
import numpy as np
import neat
import pickle

with open("winnersetup.p", "rb") as f:
    network = pickle.load(f)
    network = neat.nn.FeedForwardNetwork.create(network[0], network[1])

fileName = 'stock_data\Trade_history_A2M_03Feb2021.csv'
stock = np.array(pd.read_csv(fileName, sep=','))
stock = stock[1:127, 6:7]
print(stock)

stock = stock / (np.amax(stock))
output = network.activate(stock)
print(output)
