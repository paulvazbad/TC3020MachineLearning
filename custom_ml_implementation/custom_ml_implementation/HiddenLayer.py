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
        self.current_delta = 0

    def connect(self, number_of_previous_layer_outputs):
        '''
        Connects with a previous layer
        Must be called before any operation
        '''
        bias_neuron = 1
        self.neurons = np.ones(
            (self.number_of_neurons, number_of_previous_layer_outputs + bias_neuron))

    def get_number_of_neurons(self):
        return self.number_of_neurons

    def set_neurons(self, neurons):
        self.neurons = neurons

    def set_a(self,a):
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
        self.a = [(1 / (1+math.exp(-(z)))) for z in z_s]
        return self.a

    def calculate_error(self, delta_of_next_layer, weights_of_next_layer):
        assert len(delta_of_next_layer)==self.number_of_neurons
        neurons_without_bias = np.delete(weights_of_next_layer,0,1)
        almost = np.matmul(np.transpose(neurons_without_bias), delta_of_next_layer)
        self.current_delta =  almost * (self.a * (np.ones(self.number_of_neurons) - self.a))
        return self.current_delta