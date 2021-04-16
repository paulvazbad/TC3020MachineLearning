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
        self.neurons = np.random.rand(self.number_of_neurons, number_of_previous_layer_outputs + bias_neuron)
        print(self.neurons)
        self.delta = np.zeros((self.number_of_neurons, number_of_previous_layer_outputs)) # No bias in delta

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
        self.a = np.array([(1 / (1+math.exp(-(z)))) for z in z_s])
        return self.a

    def calculate_error(self, sigma_of_next_layer, weights_of_next_layer):
        assert len(sigma_of_next_layer)==self.number_of_neurons
        neurons_without_bias = np.delete(weights_of_next_layer,0,1)
        almost = np.matmul(np.transpose(neurons_without_bias), sigma_of_next_layer)
        self.current_sigma =  almost * (self.a * (np.ones(self.number_of_neurons) - self.a))
        return self.current_sigma
    
    def calculate_and_set_delta(self,a_of_previous_layer):
        '''
        TODO: Fix this, it should use a of the previous layer and sigma of the current one
        '''
        neurons_shape = self.neurons.shape
        print(neurons_shape)
        for i in range (0, neurons_shape[0]):
            for j in range(0,neurons_shape[1]):
                print("%d %d"%(i,j))