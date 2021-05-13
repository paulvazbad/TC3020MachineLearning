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

    def test_sample(self):
        kmeans = Kmeans()
        K = 2
        samples = kmeans.sample(self.examples_3_by_3,K)
        print(samples)
        self.assertEqual(len(samples),K)
    
    def test_calculate_distance_between_two_points(self):
        kmeans = Kmeans()
        point_a = np.array([2,3])
        point_b = np.array([1,4])
        expected_dist = (point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2 
        dist = kmeans.distance_between_two_points(point_a,point_b)
        self.assertEqual(dist,expected_dist)

    def test_calculate_closest_cluster(self):
        kmeans = Kmeans()
        example = np.array([0,0,0])
        print(self.examples_3_by_3)
        closest_cluster = kmeans.calculate_closest_cluster(example,self.examples_3_by_3)
        self.assertEqual(closest_cluster,0)

        # Now a point ver far away from the origin
        example = np.array([4,5,9]) # 9 in z
        closest_cluster = kmeans.calculate_closest_cluster(example,self.examples_3_by_3)
        self.assertEqual(closest_cluster,2)