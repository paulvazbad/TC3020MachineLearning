import unittest
from custom_ml_implementation.LogisticRegression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.logistic_regression = LogisticRegression()

    def test_h(self):
        # -3 +1 +1 = g(-1) = 1 / (1 + e^1)
        self.assertEqual(self.logistic_regression.h(
            [1, 1], [-3, 1, 1]), 0.2689414213699951)
        # Must be y= 0.5 when x1 +x2  = 3
        self.assertEqual(self.logistic_regression.h([1, 2], [-3, 1, 1]), 0.5)
        self.assertLessEqual(self.logistic_regression.h(
            [1, 1], [-3, 1, 1]), 0.5, "Error in h function")
        self.assertGreaterEqual(self.logistic_regression.h(
            [1, 4], [-3, 1, 1]), 0.5, "Error in h function")

    def test_J(self):
        self.assertAlmostEqual(self.logistic_regression.J(
            [[2], [3]], [0, 1], [1, 1]), 1.5333, 3, "Error in J Function")
        self.assertAlmostEqual(self.logistic_regression.J(
            [[0], [1]], [0, 1], [1, 1]), 0.72009, 3, "Error in J Function")

    def test_J_modified(self):
        self.assertAlmostEqual(self.logistic_regression.J_modified(
            [[0], [1]], [0, 1], [1, 1], 1), -0.059601461, 5)

    def test_train(self):
        train_x = [[2], [2], [2], [2], [6], [6], [6.6], [6.7]]
        train_y = [0, 0, 0, 0, 1, 1, 1, 1]
        INITIAL_ERROR = self.logistic_regression.J(train_x, train_y, [0, 0])
        self.logistic_regression.LEARNING_RATE = 0.5
        self.logistic_regression.MAX_ITER = 5000
        resulting_theta = self.logistic_regression.train(train_x, train_y)
        test_y = [0, 0, 0, 0, 1, 1, 1, 1]
        test_x = [[2], [2], [2], [2], [6], [6], [6.6], [6.7]]
        self.assertLessEqual(self.logistic_regression.J(
            train_x, train_y, resulting_theta), INITIAL_ERROR)

        # Validate thst classifier works
        test_y = [0, 0, 0, 0, 0, 1, 1, 1, 1]
        test_x = [[2], [2], [1], [1], [3], [5], [6], [7], [6.7]]
        for index, test in enumerate(test_x):
            pred = self.logistic_regression.h(test, resulting_theta)
            self.assertEqual(
                round(pred), test_y[index], "Wrong prediction after training :/")

    def test_train_gamma(self):
        train_x = [[2], [2], [2], [2], [6], [6], [6.6], [6.7]]
        train_y = [0, 0, 0, 0, 1, 1, 1, 1]
        INITIAL_ERROR = self.logistic_regression.J(train_x, train_y, [0, 0])
        self.logistic_regression.LEARNING_RATE = 0.5
        self.logistic_regression.MAX_ITER = 5000
        # Here is gamma
        self.gamma = 1
        resulting_theta = self.logistic_regression.train(train_x, train_y)
        test_y = [0, 0, 0, 0, 1, 1, 1, 1]
        test_x = [[2], [2], [2], [2], [6], [6], [6.6], [6.7]]
        self.assertLessEqual(self.logistic_regression.J(
            train_x, train_y, resulting_theta), INITIAL_ERROR)

        # Validate thst classifier works
        test_y = [0,0,0,0,0,1,1,1,1]
        test_x = [[2],[2],[1],[1],[3],[5],[6],[7],[6.7]]
        for index,test in enumerate(test_x):
            pred = self.logistic_regression.h(test,resulting_theta)
            self.assertEqual(round(pred),test_y[index], "Wrong prediction after training :/")

if __name__ == '__main__':
    unittest.main()
