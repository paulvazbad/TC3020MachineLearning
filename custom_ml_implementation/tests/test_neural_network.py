import unittest
from custom_ml_implementation.NeuralNetwork import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):

    def test_init(self):
        nn = NeuralNetwork(3, 3)
        self.assertEqual(len(nn.layers), 2)  # Input and output layer

    def test_connection(self):
        '''
        Verify the shapes of input and output neurons
        '''
        ann = NeuralNetwork(3, 3)
        self.assertEqual(len(ann.layers), 2)  # Input and output layer

        # First and second layers must generate 4 outputs
        self.assertEqual(len(ann.layers[0].activation_function([3, 3, 3])), 3)

        # Generates 3 outputs
        self.assertEqual(ann.layers[1].get_number_of_neurons(), 3)

    def test_feed_forward(self):
        print('Feed forward test')


if __name__ == '__main__':
    unittest.main()
