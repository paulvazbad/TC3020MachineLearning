from .LinearRegression import LinearRegression

class LogisticRegression(LinearRegression):
    '''Implementation of the Logistic Regression cost function, hypothesis, etc... '''
    def __init__(self,LEARNING_RATE = 0.005,TARGET_ERROR = 0.001,MAX_ITER= 5000):
        self.LEARNING_RATE = LEARNING_RATE
        self.TARGET_ERROR = TARGET_ERROR
        self.MAX_ITER = MAX_ITER

