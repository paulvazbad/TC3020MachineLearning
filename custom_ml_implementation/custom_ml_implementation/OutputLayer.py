import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .HiddenLayer import HiddenLayer


class OutputLayer(HiddenLayer):
    '''
    Output of an Artificial Neural Network
    delta is calculated differently
    '''
    def __init__(self, number_of_neurons=2):
        super().__init__(number_of_neurons)

    # Operations
    def calculate_error(self, outputs):
        # Returns the same input but with bias neuron
        return self.a - np.array(outputs)
