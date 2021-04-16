import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .HiddenLayer import HiddenLayer 

class InputLayer(HiddenLayer):
    '''
    Input of an Artificial Neural Network
    resulting z is the same as the input
    '''

    def __init__(self, number_of_inputs=2):
        '''
        Default number of neurons per layer is 2
        '''
        self.number_of_neurons = number_of_inputs

    # Operations
    def activation_function(self, inputs):
        # Returns the same input but with bias neuron
        self.a = inputs
        return inputs
