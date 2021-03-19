import unittest
from custom_ml_implementation.LinearRegression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.linear_regression = LinearRegression()

    def test_h(self):
        self.assertEqual(self.linear_regression.h([2],[1,3]), 7,"Error in h function")
        self.assertEqual(self.linear_regression.h([3],[1,3]), 10, "Error in h function")

    def test_J(self):
        self.assertEqual(self.linear_regression.J([[2],[3]],[3,4],[1,1]), 0.0, "Error in J Function")
        self.assertEqual(self.linear_regression.J([[2],[3]],[3,4],[1,0.5]), 0.8125, "Error in J Function")
        self.assertEqual(self.linear_regression.J([[1],[2],[3]],[1,2,3],[0,0]), 2.3333333333333335, "Error in J Function")

    def test_J_modified(self):
        self.assertEqual(self.linear_regression.J_modified([[2],[3]],[3,4],[1,0.5], 1),-3.25)
        self.assertAlmostEqual(self.linear_regression.J_modified([[1],[2],[3]],[1,2,3],[0,0], 1),-4.666666666666667)
    
    def test_train(self):
        train_x = [[0],[1],[2],[3],[4],[5],[6],[7]]
        train_y = [2,3,4,5,6,7,8,9]
        resulting_theta = self.linear_regression.train(train_x,train_y)
        test_y = [10,11,12,13,14,15,16,17]
        test_x = [[8],[9],[10],[11],[12],[13],[14],[15]]
        self.assertLessEqual(self.linear_regression.J(train_x,train_y,resulting_theta), self.linear_regression.TARGET_ERROR)
       

if __name__ == '__main__':
    unittest.main()