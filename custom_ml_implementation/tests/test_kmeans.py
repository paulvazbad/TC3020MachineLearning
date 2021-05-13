import unittest
import numpy as np
from custom_ml_implementation.Kmeans import Kmeans


class TestKmeans(unittest.TestCase):
    def setUp(self) -> None:
        self.examples_3_by_2 = np.arange(6).reshape((3,2))
        self.examples_3_by_3 = np.arange(9).reshape((3,3))
        
    def test_average_of_points(self):
        kmeans = Kmeans()
        self.assertSequenceEqual(kmeans.average_of_points(self.examples_3_by_3).tolist(),[3,4,5])
        print("test_fit")