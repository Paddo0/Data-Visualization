from __future__ import print_function
import pandas as pd
import os
import neat
import pickle

train_labels = pd.read_csv('trainingLabels.csv', sep=',', header=None)
train_price = pd.read_csv('trainingData.csv', sep=',', header=None)
train_labels = train_labels.values.tolist()
train_price = train_price.values.tolist()
print(len(train_labels))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Base Fitness
        genome.fitness = 1
        # Create Network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(train_price, train_labels):
            output = net.activate(xi)
            genome.fitness -= ((output[0] - xo[0]) ** 2) / (len(train_labels))


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(15))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 30)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open('winnersetupOHLCv2.p', 'wb') as f:
        pickle.dump([winner, config], f)
    for xi, xo in zip(train_price, train_labels):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
