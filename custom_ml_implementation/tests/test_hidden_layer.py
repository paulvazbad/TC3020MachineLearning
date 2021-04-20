import unittest
import numpy as np
from custom_ml_implementation.HiddenLayer import HiddenLayer


class TestHiddenLayer(unittest.TestCase):
    def setUp(self):
        self.hidden_layer = HiddenLayer(3)
        self.hidden_layer.connect(3)
        self.hidden_layer.set_neurons(
            np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]))

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
            a).tolist(), [0.9820137900379085, 0.9820137900379085, 0.9820137900379085])

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

    def test_error_function(self):
        hidden_layer = HiddenLayer(2)
        output_layer = HiddenLayer(2)
        output_layer.connect(2)
        # Test example seen in class
        hidden_layer.set_a([0.593269992, 0.596884378])

        # Set weights of the last layer
        output_layer.set_neurons(
            np.array([[0.60, 0.4, 0.45], [0.60, 0.50, 0.55]]))
        self.assertSequenceEqual(hidden_layer.calculate_error(
            [0.741365069, -0.217071535], output_layer.neurons).tolist(), [0.045367008984756374, 0.05154539411422879])

    def test_calculate_and_set_delta(self):
        '''
        Verifies the shape of the deltas equals the shape of the weights
        '''

       # layer with 2 neurons
        output_layer = HiddenLayer(2)
        # mock a previous layer with 3 neurons
        output_layer.connect(2)
        # Example seen in class (slide 14)
        output_layer.current_sigma = np.array([0.741365069, -0.217071535])
        output_layer.calculate_and_set_delta([0.593269992, 0.596884378])
        self.assertSequenceEqual(output_layer.delta[:, 0].tolist(), [
                                  0.741365069, -0.217071535])

        self.assertEquals(output_layer.delta.shape, output_layer.neurons.shape)

    def test_update_weigths_with_deltas(self):
        # layer with 2 neurons
        output_layer = HiddenLayer(2)
        # mock a previous layer with 3 neurons
        output_layer.connect(2)
        output_layer.calculate_and_set_delta([1, 1, 1])
        self.assertEquals(output_layer.delta.shape, output_layer.neurons.shape)

        # Test that the reg factor is considered correctly
        number_of_examples_used = 1
        learning_rate = 0.5
        reg_factor = 0.0
        output_layer.current_sigma = np.array([0.741365069, -0.217071535])
        output_layer.calculate_and_set_delta([0.593269992, 0.596884378])
        output_layer.set_neurons(
            np.array([[0.60, 0.4, 0.45], [0.60, 0.50, 0.55]])) 
        output_layer.update_weights_with_deltas(
            number_of_examples_used, learning_rate, reg_factor)
        output_layer.print_weights()

if __name__ == '__main__':
    unittest.main()
