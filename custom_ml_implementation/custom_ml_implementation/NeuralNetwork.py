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
                

    def addHiddenLayer(self, number_of_neurons):
        self.layers.insert(len(self.layers) - 1,HiddenLayer(number_of_neurons))
        self.connect_all_layers()


    def feed_forward(self,input_data):
        a = input_data
        for index,layer in enumerate(self.layers):
            a = layer.activation_function(a)
            #print("output layer_%d "%(index))
            #print(a)
        return a