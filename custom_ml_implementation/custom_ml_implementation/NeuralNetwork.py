import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .HiddenLayer import HiddenLayer
from .InputLayer import InputLayer
from .OutputLayer import OutputLayer


class NeuralNetwork():

    def __init__(self, input_layer_size=2, output_layer_size=2):
        self.layers = []
        self.layers.append(InputLayer(input_layer_size))  # InputLayer
        self.layers.append(OutputLayer(output_layer_size))  # OutputLayer
        self.connect_all_layers()

    def connect_all_layers(self):
        '''
        Connects all layers in the network
        Warning: This erases learned weights
        '''
        for index, layer in enumerate(self.layers):
            if(index > 0):
                self.layers[index].connect(
                    self.layers[index - 1].get_number_of_neurons())

    def addHiddenLayer(self, number_of_neurons):
        self.layers.insert(len(self.layers) - 1,
                           HiddenLayer(number_of_neurons))
        self.connect_all_layers()

    def feed_forward(self, input_data):
        a = input_data
        for index, layer in enumerate(self.layers):
            a = layer.activation_function(a)
            #print("output layer_%d "%(index))
            # print(a)
        return a

    def update_deltas(self, output_example):
        num_of_layers = len(self.layers)
        i = num_of_layers - 1
        # Calculate the sigma of the output layer
        sigma_of_next_layer = self.layers[i].calculate_error(output_example)
        weights_of_next_layer = self.layers[i].neurons
        self.layers[i].calculate_and_set_delta(self.layers[i - 1].a)
        i -= 1
        while(i > 0):
            self.layers[i].calculate_error(
                sigma_of_next_layer, weights_of_next_layer)

            # Also update the delta of this layer
            self.layers[i].calculate_and_set_delta(self.layers[i - 1].a)

            weights_of_next_layer = self.layers[i].neurons
            sigma_of_next_layer = self.layers[i].current_sigma
            i-=1

    def backpropagation(self, input_examples, output_example):
        for example in input_examples:
            self.feed_forward(example)
            self.update_deltas(output_example)

    # Access for testing
    def output_layer(self):
        return self.layers[-1]
