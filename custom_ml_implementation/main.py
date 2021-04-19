from custom_ml_implementation.LinearRegression import LinearRegression
from custom_ml_implementation.LogisticRegression import LogisticRegression
from custom_ml_implementation.NeuralNetwork import NeuralNetwork
import pandas as pd
import numpy as np


def main():
    # Test the XOR dataset
    df = pd.read_csv('./custom_ml_implementation/datasets/xor.csv', sep=',')
    features = df[['x', 'y']].values
    print(features[0])
    labels = df['label'].values
    print(labels[0])
    EPOCHS = 100
    LEARNING_RATE = 0.05
    REGULARIZATION = 0.5
    nn = NeuralNetwork(2, 1)
    nn.addHiddenLayer(2)
    nn.train(np.array(features), np.array(labels),
             LEARNING_RATE, REGULARIZATION, EPOCHS)
    print(nn.feed_forward([0.01, 0.01]), "Expected 0")
    print(nn.feed_forward([0.01, 1.0]), "Expected 1")
    print(nn.feed_forward([1.0, 0.01]), "Expected 1")
    print(nn.feed_forward([1.0, 1.0]), "Expected 0")


if __name__ == "__main__":
    main()
