import unittest
import numpy as np

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

    def test_adding_hidden_layers(self):
        '''
        Verify that adding hidden layers work
        '''
        ann = NeuralNetwork(3, 3)
        ann.addHiddenLayer(2)
        expected_shape = [3, 2, 3]
        for index, layer in enumerate(ann.layers):
            self.assertEquals(layer.get_number_of_neurons(),
                              expected_shape[index])

    def test_feed_forward(self):
        print('Feed forward test')
        nn = NeuralNetwork(2, 1)  # 2 inputs, 1 output
        nn.addHiddenLayer(2)
        # Modify the weights
        layer_1_weights = np.array([[-30, 20, 20], [10, -20, -20]])
        nn.layers[1].set_neurons(layer_1_weights)
        output_layer_weights = np.array([[-10, 20, 20]])
        nn.layers[2].set_neurons(output_layer_weights)

        for x1 in range(0, 2):
            for x2 in range(0, 2):
                output = round(nn.feed_forward([x1, x2])[0])
                #print('x1: %d x2 : %d output: %d'%(x1,x2,output))
                self.assertEquals(output, int(not(bool(x1) != bool(x2))))


    def test_backpropagation(self):
        nn = NeuralNetwork(2,2)
        nn.addHiddenLayer(2)
        # Example seen in class
        nn.output_layer().set_neurons(np.array([[0.60,0.4,0.45],[0.60,0.50,0.55]]))
        nn.output_layer().print_weights()
        nn.layers[1].set_neurons(np.array([[0.35,0.15,0.20],[0.35,0.25,0.30]]))
        nn.layers[1].print_weights()
        input_examples = [[0.05,0.10]]
        ouput_example = [0.01,0.99]
        nn.backpropagation(input_examples,ouput_example)

if __name__ == '__main__':
    unittest.main()
