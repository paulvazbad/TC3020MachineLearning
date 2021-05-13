import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self):
        print("init")

    def fit(self):
        '''
        Have afitmethod (it might use other methods/funcions) that will receive the dataset and the number of
        clusters (k) that must be found,  number of iterations must be received as well (maxiter).  This method
        should reply with:
        - A list with thecoordinatesof each centroid (theu1, . . . uk).
        - A list with the index of the assigned centroids for each example (thec(i)).
        '''
        print("Fit")

    def predict(self):
        '''
        Calculates what would be the cluster to be assigned to a list of examples. 
        Receives a list of examples (with the same number of features as in thefitfunction); you can think of thislist of examples as a matrix with examples and their features.  
        This function should obtain the list of thepredicted centroids for each example (thec(i)).
        '''
        print("predict")

    def elbow(self):
        '''
       Create the plot to see how the cost function changes depending on the selectedkvalue. (testingwithk= 2. . .7is just fine).
        '''
        print("elbow")