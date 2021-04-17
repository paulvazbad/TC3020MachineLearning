import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HiddenLayer():
    '''
    HiddenLayer of an Artificial Neural Network
    Has one main attribute:
        - Neurons (each represented as a vector of 1 dimensional weights)
    Can perform several operations on them:
    '''

    def __init__(self, number_of_neurons=2):
        '''
        Default number of neurons per layer is 2

        '''
        self.number_of_neurons = number_of_neurons
        self.a = []
        self.current_sigma = []

    def connect(self, number_of_previous_layer_outputs):
        '''
        Connects with a previous layer
        Must be called before any operation
        '''
        bias_neuron = 1
        self.neurons = np.random.rand(
            self.number_of_neurons, number_of_previous_layer_outputs + bias_neuron)
        # print(self.neurons)
        # Experiment: bias also has delta
        self.delta = np.zeros(self.neurons.shape)

    def get_number_of_neurons(self):
        return self.number_of_neurons

    def set_neurons(self, neurons):
        self.neurons = neurons

    def set_a(self, a):
        self.a = a

    def size_of_weights(self):
        return (self.neurons.shape)

    def print_weights(self):
        print(self.neurons)

    # Operations
    def z(self, inputs):
        inputs_with_bias = np.insert(inputs, 0, 1)
        return np.matmul(self.neurons, inputs_with_bias)

    def activation_function(self, inputs):
        z_s = self.z(inputs)
        self.a = np.array([(1 / (1+math.exp(-(z)))) for z in z_s])
        return self.a

    def calculate_error(self, sigma_of_next_layer, weights_of_next_layer):
        #assert len(sigma_of_next_layer) == self.number_of_neurons
        neurons_without_bias = np.delete(weights_of_next_layer, 0, 1)
        g = (self.a * (np.ones(self.number_of_neurons) - self.a))
        self.current_sigma = np.matmul(np.transpose(
            neurons_without_bias), sigma_of_next_layer) * g
        return self.current_sigma

    def calculate_and_set_delta(self, a_of_previous_layer):
        '''
        TODO: implement this
        '''
        #print(self.current_sigma * np.transpose(a_of_previous_layer))
        #new_delta = self.delta + self.current_sigma * np.transpose(a_of_previous_layer)
        # print(new_delta)
        # Insert BIAS into a of previous layer so we can get the delta of the bias in this layer
        a_of_previous_layer_with_bias = np.insert(a_of_previous_layer, 0, 1)
        for i in range(0, len(self.current_sigma)):
            for j in range(0, len(a_of_previous_layer_with_bias)):
                # Generate delta of weight
                self.delta[i][j] = a_of_previous_layer_with_bias[j] * \
                    self.current_sigma[i]
