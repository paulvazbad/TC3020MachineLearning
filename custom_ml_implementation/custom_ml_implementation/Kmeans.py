import math
import random

import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans():
    def __init__(self):
        self.MAX_ITER = 100

    def fit(self, K=0, examples=np.ndarray((1, 1)), max_iter=100):
        '''
        Have afitmethod (it might use other methods/funcions) that will receive the dataset and the number of
        clusters (k) that must be found,  number of iterations must be received as well (maxiter).  This method
        should reply with:
        - A list with thecoordinatesof each centroid (theu1, . . . uk).
        - A list with the index of the assigned centroids for each example (thec(i)).
        '''
        print("Fit")
        self.MAX_ITER = max_iter
        # If example is N-Dimensional a centroid must also be N-Dimensional
        # Initialize
        K_coords = self.sample(examples, K)
        print('Intiial coords (random)')
        print(K_coords)

        # For max_iter
        for i in range(0, self.MAX_ITER):
            print("---EPOCH: %d---" % i)
            # C
            examples_grouped_by_cluster = []
            
            for i in range(0,K):
                examples_grouped_by_cluster.append([])

            # Calculate to which cluster an example x(i) belongs to
            # C[K] = [index_x1, index_x2,..., index_xn]
            for index, example in enumerate(examples):
                closest_cluster = self.calculate_closest_cluster(
                    example, K_coords)
                # Append the example in dataset of the example
                examples_grouped_by_cluster[closest_cluster].append(index)
            
            #cost = self.J(K_coords, examples_grouped_by_cluster)    
            #print("Error:" + str(cost))
            
            # Move the centroid of the cluster to the average of the assigned points
            for index, examples_in_cluster in enumerate(examples_grouped_by_cluster):
                print("Examples assigned to cluster %d"%index)
                print(examples[examples_in_cluster,:])
                K_coords[index] = self.average_of_points(examples[examples_in_cluster,:])

            print("New coords: ")
            print(K_coords)
            
            

    def average_of_points(self, examples_grouped_by_cluster):
        '''
        [[0,1,2]
        [3,4,5]
        [6,7,8]]
        returns [3,4,5]
        '''
        return average(examples_grouped_by_cluster, axis=0)

    def sample(self, array_of_arrays, K):
        '''
        [[0,1,2]
        [3,4,5]
        [6,7,8]] and K=1
        returns one row
        '''
        sampled_indexes = []
        samples = []
        while(len(samples) < K):
            rand_index = random.randint(0, len(array_of_arrays) - 1)
            if(rand_index not in sampled_indexes):
                samples.append(array_of_arrays[rand_index])
                sampled_indexes.append(rand_index)
        return np.array(samples)

    def calculate_closest_cluster(self, example, K_coords):
        '''
        example = [f1,f2,f3]
        K_coords = [[c_0,c_1,c_3],[c_0,c_1,c_3],[c_0,c_1,c_3]]
        returns the index of K_coords that is closest
        '''
        closest_distance = self.distance_between_two_points(
            example, K_coords[0])
        closest_cluster = 0
        for index, K_coord in enumerate(K_coords):
            distance_to_cluster_center = self.distance_between_two_points(
                example, K_coord)
            if(distance_to_cluster_center < closest_distance):
                closest_distance = distance_to_cluster_center
                closest_cluster = index
        return closest_cluster

    def distance_between_two_points(self, point_a, point_b):
        '''
        Calculate euclidean distance between two points (squared)
        point_a = [x_a,y_a]
        point_b = [x_b,y_b]
        returns square(x_a - x_b) + square(y_a - y_b)

        '''
        return np.sum(np.square(point_a-point_b))

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

    def J(self, K_coords, examples_grouped_by_cluster):
        '''

        CHECK THIS
        examples_grouped_by_cluster[K][features] = [[index_x1, index_x2,..., index_xn]]
        '''
        total_cost = 0
        for cluster_number, cluster_examples in enumerate(examples_grouped_by_cluster):
            for example in cluster_examples:
                total_cost += self.distance_between_two_points(
                    K_coords[cluster_number], example)
        return total_cost
