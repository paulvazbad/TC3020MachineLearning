from custom_ml_implementation.LinearRegression import LinearRegression
from custom_ml_implementation.LogisticRegression import LogisticRegression
from custom_ml_implementation.NeuralNetwork import NeuralNetwork
from custom_ml_implementation.Kmeans import Kmeans
import pandas as pd
import numpy as np

train_or_load = 'load'

def main():
    group_a = np.arange(12).reshape((6,2)) #0-11
    group_b = np.arange(500,512).reshape((6,2)) #50-61
    examples  = np.concatenate((group_a, group_b), axis=0)
    examples = np.array(examples)
    kmeans = Kmeans()
    kmeans.fit(2,examples,10)
    kmeans.plot_clusters(examples)


if __name__ == "__main__":
    main()
