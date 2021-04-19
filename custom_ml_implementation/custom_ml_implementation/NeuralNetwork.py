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
        self.learning_rate = 0
        self.reg_factor = 0

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

    def feed_forward(self, input_data, update_weights=False, number_of_examples_used=1):
        a = input_data
        for index, layer in enumerate(self.layers):
            a = layer.activation_function(a)
            # weights only available in HiddenLayer and OutputLayer
            if(update_weights and index > 0):
                layer.update_weights_with_deltas(
                    number_of_examples_used, self.learning_rate, self.reg_factor)
            # print("output layer_%d "%(index))
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
            # Since the weights belong to the layer they GO TO not the layer
            # they COME FROM
            # we dont do: delta(l) = delta(l) + a(l)error(l+1)
            # we must use: delta(l) = delta(l) + a(l-1)error(l)
            self.layers[i].calculate_and_set_delta(self.layers[i - 1].a)

            weights_of_next_layer = self.layers[i].neurons
            sigma_of_next_layer = self.layers[i].current_sigma
            i -= 1

    def backpropagation(self, input_examples, output_examples):
        assert(len(input_examples) == len(output_examples))
        # First iteration dont update weights yet
        update_weights = False
        for index, example in enumerate(input_examples):
            # print("Backpropagation with example %d" % (index))
            self.feed_forward(example, update_weights, len(input_examples))
            self.update_deltas(output_examples[index])
            update_weights = True

        # Print error in last layer after this backpropagation iteration
        print("Cost: ")
        print(self.cost_function(input_examples, output_examples))

    def train(self, input_examples, output_examples, learning_rate, reg_factor, epochs):
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
        for i in range(0, epochs):
            print("---EPOCH: %d---" % i)
            self.backpropagation(input_examples, output_examples)

    def predict(self, input_data):
        a = input_data
        for index, layer in enumerate(self.layers):
            a = layer.activation_function(a)
        return a

    def print_weights(self):
        for index, layer in enumerate(self.layers):
            if(index > 0):
                print("Weights of layer %d" % index)
                layer.print_weights()

    def cost_function(self, input_examples, output_examples):
        total_cost = 0
        m = len(input_examples)
        for index, example in enumerate(input_examples):
            h_s = self.predict(input_examples[index])
            y_ks = [output_examples[index]]

            for index,y_k in enumerate(y_ks):
                h = h_s[index]
                y_k = y_ks[index]
                total_cost+= y_k*math.log(h) + (1 - y_k)*math.log(1-h)
        
        total_cost = (-1/m) * total_cost
        #print("Total cost without regul")
        #print(total_cost)
        # Regularization
        neurons_sum = 0
        for index, layer in enumerate(self.layers):
            if(index > 0):
                for neuron in layer.neurons:
                    neurons_sum+= np.sum(np.square(neuron))

        return total_cost + (self.reg_factor/(2*m))*neurons_sum
    # Access for testing

    def output_layer(self):
        return self.layers[-1]
