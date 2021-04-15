import unittest
import numpy as np
from custom_ml_implementation.HiddenLayer import HiddenLayer


class TestHiddenLayer(unittest.TestCase):
    def setUp(self):
        self.hidden_layer = HiddenLayer(3)
        self.hidden_layer.connect(3)

    def test_correct_weight_matrix_shape(self):
        previous_layer_number_neurons = 3
        for i in range(1, 6):
            self.hidden_layer = HiddenLayer(i)
            self.hidden_layer.connect(previous_layer_number_neurons)
            self.assertEqual(self.hidden_layer.size_of_weights()[
                             0], i)  # number of neurons
            self.assertEqual(self.hidden_layer.size_of_weights()[
                             1], previous_layer_number_neurons + 1)  # and bias
            previous_layer_number_neurons += 1

    def test_z(self):
        '''
        Tests the matrix multiplication
        '''
        a = np.arange(3)
        print(a)
        # Output number same as neuron number
        self.assertEquals(len(self.hidden_layer.z(a)),
                          self.hidden_layer.get_number_of_neurons())
        # Output is correct
        self.assertSequenceEqual(
            self.hidden_layer.z(a).tolist(), [4.0, 4.0, 4.0])
        print("Tests z of a hidden layer")

    def test_activation_function(self):
        a = np.arange(3)
        print(a)
        self.assertSequenceEqual(self.hidden_layer.activation_function(
            a), [0.9820137900379085, 0.9820137900379085, 0.9820137900379085])

    def test_AND_example(self):
        hidden_layer = HiddenLayer(1)
        hidden_layer.connect(2)
        hidden_layer.print_weights()
        # Set weights to the ones validated in class
        hidden_layer.set_neurons(np.array([[-30, 20, 20]]))
        hidden_layer.print_weights()
        for x1 in range(0, 2):
            for x2 in range(0, 2):
                output = round(hidden_layer.activation_function([x1, x2])[0])
                print('x1: %d x2 : %d output: %d' % (x1, x2, output))
                self.assertEquals(output, int(x1 and x2))


if __name__ == '__main__':
    unittest.main()
