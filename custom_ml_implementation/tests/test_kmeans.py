import unittest
import numpy as np
from custom_ml_implementation.Kmeans import Kmeans


class TestHiddenLayer(unittest.TestCase):
    def test_fit(self):
        kmeans = Kmeans()
        kmeans.fit()
        print("test_fit")