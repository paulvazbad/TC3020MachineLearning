import unittest
import numpy as np
from custom_ml_implementation.HiddenLayer import HiddenLayer

class TestHiddenLayer(unittest.TestCase):
    def setUp(self):
        self.hidden_layer = HiddenLayer(3)
        self.hidden_layer.connect(3)

    def test_correct_weight_matrix_shape(self):
        previous_layer_number_neurons = 3
        for i in range(1,5):
            self.hidden_layer = HiddenLayer(i)
            self.hidden_layer.connect(previous_layer_number_neurons)
            self.assertEqual(self.hidden_layer.size_of_weights()[0],i) # number of neurons
            self.assertEqual(self.hidden_layer.size_of_weights()[1],previous_layer_number_neurons + 1) # and bias

    def test_z(self):
        '''
        TODO: Decide if bias from input layer
        or append to each hidden layer a bias in the first position
        '''
        a = np.arange(3)
        print(a)
        print(self.hidden_layer.print_weights())
        print("Tests z of a hidden layer")

    def test_activation_function(self):
       print("tests activation function of a hidden layer")

if __name__ == '__main__':
    unittest.main()