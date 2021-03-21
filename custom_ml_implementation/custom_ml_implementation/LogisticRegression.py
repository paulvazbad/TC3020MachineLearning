import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .LinearRegression import LinearRegression


class LogisticRegression(LinearRegression):
    '''Implementation of the Logistic Regression cost function, hypothesis, etc... '''

    def __init__(self, LEARNING_RATE=0.005, TARGET_ERROR=0.001, MAX_ITER=5000):
        self.LEARNING_RATE = LEARNING_RATE
        self.TARGET_ERROR = TARGET_ERROR
        self.MAX_ITER = MAX_ITER

    def h(self, x, theta_values):
        return(1 / (1+math.exp(- (super().h(x, theta_values)))))

    def J(self, x, y, theta_values):
        assert len(x) == len(y), "Length of X is not equal to length of Y"
        total_sum = 0
        m = len(x)
        for i in range(0, len(x)):
            total_sum += (y[i] * math.log(self.h(x[i], theta_values))) + \
                          ((1 - y[i]) * math.log(1 - self.h(x[i], theta_values)))

        cost = -total_sum/(m)
        if not np.isfinite(cost):
            print("Cost generated is infinite")
            print("When: ")
            print("Theta values: " + str(theta_values))
            print("h(x) " + str(h(x[i], theta_values)))
            raise Exception("RIP")
        return cost

    def J_modified(self,x,y,theta_values, feature_to_multiply):
        '''
        Given a set of theta_values calculate te cost of the hypothesis h(theta) times x(i)?
        FOR UNIVARIABLE LINEAR REGRESSION
        '''

        assert len(x)==len(y),"Length of X is not equal to length of Y"+str(len(x)) + " vs " +str(len(y))
        assert feature_to_multiply <= len(x), "Feature "+str(feature_to_multiply) + "doesnt exist in array of size: " + str(len(x))
        total_sum = 0
        m = len(x)
        temp=0
        for i in range(0,len(x)):
            temp=(self.h(x[i],theta_values) - y[i])
            val_to_multiply = 1
            if(feature_to_multiply > 0):
                val_to_multiply = x[i][feature_to_multiply - 1]
            temp = temp * val_to_multiply
            total_sum+=temp

        return total_sum/(m)

    def predict(self,input_value,theta_values):
        return round(h(input_value, resulting_theta))
