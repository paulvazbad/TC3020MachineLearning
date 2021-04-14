import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .HiddenLayer import HiddenLayer
from .InputLayer import InputLayer

class NeuralNetwork():
   
    def __init__(self,input_layer_size=2,output_layer_size=2):
        self.layers = []
        self.layers.append(InputLayer(input_layer_size)) # InputLayer
        self.layers.append(HiddenLayer(output_layer_size)) # OutputLayer
        self.connect_all_layers()

    def connect_all_layers(self):
        '''
        Connects all layers in the network
        Warning: This erases learned weights
        '''
        for index,layer in enumerate(self.layers):
            if(index > 0):
                self.layers[index].connect(self.layers[index - 1].get_number_of_neurons())
                
