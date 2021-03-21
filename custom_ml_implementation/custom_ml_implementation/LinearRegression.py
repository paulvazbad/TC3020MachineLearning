import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""# Implementation of the linear regression, gradient descend, hypothesis, etc..."""
class LinearRegression():  
  def __init__(self,LEARNING_RATE = 0.005,TARGET_ERROR = 0.001,MAX_ITER= 5000):
    self.LEARNING_RATE = LEARNING_RATE
    self.TARGET_ERROR = TARGET_ERROR
    self.MAX_ITER = MAX_ITER

  def h(self,x,theta_values):
    '''
    Given features= [feature_0,feature_1, ...feature_n] and theta_values return predicted value using the hypothesis:
    h(x) = theta_0 + theta_1*feature_0 + theta_n+1 * feature_n
    '''
    assert len(theta_values)==len(x)+1, "Number of features:" +str(len(x)+1) + "invalid to match hypothesis " + str(len(theta_values))
    x_with_x0=np.insert(x,0,1)
    return np.matmul(x_with_x0,theta_values)


  def J(self,x,y,theta_values):
    '''
    Given a set of theta_values calculate te cost of the hypothesis h(theta)
    '''
    assert len(x)==len(y),"Length of X is not equal to length of Y"
    total_sum = 0
    m = len(x)
    for i in range(0,len(x)):
      total_sum+=(self.h(x[i],theta_values) - y[i])**2

    cost = total_sum/(2*m)
    if not np.isfinite(cost):
      print("Cost generated is infinite")
      print("When: ")
      print("Theta values: " + str(theta_values))
      print("h(x) " + str(h(x[i],theta_values)))
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

  def gradient_descent(self,learning_factor,target_error,max_iter,initial_theta,x,y):
    #INITIALS
    theta_values = initial_theta
    if(len(initial_theta)==0):
      theta_values = [ 0 for i in x[0]]
      theta_values.append(0)
    temp_theta_values = theta_values.copy()

    num_iter=0
    last_error = self.J(x,y,theta_values)
    error = last_error

    print('TRAINING......')
    while(num_iter<max_iter and error>target_error):
      # Calculate new theta_values
      print("-------------------")
      print("EPOCH: " + str(num_iter))
      for index,theta in enumerate(theta_values):
        print('Theta_'+str(index)+" :" + str(theta_values[index]))
      
      print("Error:" + str(error))


      # For each feature a theta should be calculated
      for index in range(0,len(x[0])+1):
        temp_theta_values[index] = theta_values[index] - learning_factor *  self.J_modified(x,y,theta_values,index)

      # Update theta_values
      theta_values = temp_theta_values.copy()

      # Check error
      error = self.J(x,y,theta_values)
      num_iter+=1

      # Check convergence?
      # If error > last_error decrease learning rate
    '''
      if error > last_error:
        print("Not converging!")
        learning_factor = learning_factor / 3
        print("New Learning factor: " + str(learning_factor))
      # if error < last_error increase learning rate
      if error < last_error:
        print("Converging!")
        learning_factor = learning_factor * 1.5
        print("New Learning factor: " + str(learning_factor))

      last_error = error
    '''
      # Feedback

    print("-------------------")

    return theta_values

  def train(self,x,y,initial_theta=[]):
    '''
    Given array inputs x_0, x_1, x_n
    Returns a vector of coefficients theta_0, theta_1, thetha_n+1 that will be used as model.
    '''
    resulting_theta = self.gradient_descent(self.LEARNING_RATE,self.TARGET_ERROR,self.MAX_ITER,initial_theta,x,y)
    print("RESULT")
    for index,theta in enumerate(theta_values):
      print('Theta_'+str(index)+" :" + str(theta_values[index]))
    
    return resulting_theta

  def get_normalized_value(self,value,mean,range):
    return (value - mean)/ (range)


  def get_real_value(self,value,mean,range):
    return (value * range) + mean

  def predict(self,input_value,resulting_theta):
    normalized_input = get_normalized_value(input_value,SALINITY_MEAN, SALINITY_RANGE)
    normalized_output = h([normalized_input], resulting_theta)
    return get_real_value(normalized_output,TEMPERATURE_MEAN,TEMPERATURE_RANGE)