import numpy as np
import pandas as pd
import scipy.special as sp

# Author: Shiva Keshav Govindaraju
# Class: COEN 420 Machine Learning
# Homework Assignment 1, Problem 3 B and C

# Please run this script so that Main will execute.

def sigmoid_G(z):
    #this is essentially h_theta(z)
    #return 1 / (1 + np.exp(-z))
    return sp.expit(z) #np.exp didn't have enough precision to handle things

def gradient_L(theta, x, y):
    z = y * x.dot(theta) #extracting this because it's long enough
    #gradient(theta) = -1 * SUM of (1-h_theta(z)) * y * xT
    g = -np.mean((1 - sigmoid_G(z)) * y * x.T, axis=1)
    return g

def hessian_H(theta, x, y):
    hessian = np.zeros((x.shape[1], x.shape[1])) #initializing the empty matrix to make things easier
    z = y * x.dot(theta) #calculating z in advance because it's constant
    m = hessian.shape[0]
    for i in range(m):
        for j in range(i, m): #Hessians are symmetric, don't bother double-computing
            if i <= j:
                # This is essentially the function I found as my answer to HW1 Problem 3A
                # H = SUM of h_theta(z) * (1 - h_theta(z)) * xi * xj
                hessian[i][j] = np.mean(sigmoid_G(z) * (1 - sigmoid_G(z)) * x[:,i] * x[:,j])
                if i != j:
                    hessian[j][i] = hessian[i][j] #The Hessian is ALWAYS a Symmetric matrix
    return hessian

def newton_raphson_regression(theta_0, x, y, tolerance):
    theta = theta_0 #saving the initial theta to be used
    diff = 100 #arbitrarily large since we plan on doing at least one step
    step = 1 # step counter
    while diff > tolerance: #we need to make steps until the next theta-change is less than the tolerance
        old_theta = theta.copy() #save the old theta so we can compute diff later
        # Occasionally in testing, H became Singular (and thus, non-invertible)
        # so this try-catch block is here just because it catches that and accounts for it
        # for this dataset, it's apparently not supposed to do that, though
        # but if the print-statement in the except block occurs, that's why.
        try:
            inverse_Hessian = np.linalg.inv(hessian_H(theta, x, y))
        except np.linalg.LinAlgError:
            print("Ran into Non-Invertible Hessian in Step {}. Trying Pseudo-Inverse...".format(step))
            inverse_Hessian = np.linalg.pinv(hessian_H(theta, x, y))
        #invH = np.linalg.inv(hessian_H(theta, x, y)) # need the inverse of the Hessian of the function
        gradient = gradient_L(theta, x, y) # need the gradient of the function
        theta = theta - inverse_Hessian.dot(gradient) # update theta := theta - H^-1 dot gradient
        diff = np.linalg.norm(theta - old_theta, ord=1) # calculate the change in theta
        step = step + 1
    return theta

def theta_steps(theta_0, x, y, tolerance):
    # This function is just for getting the thetas calcualted at each step
    theta = theta_0
    diff = 100
    step = 1
    theta_list = [theta]
    while diff > tolerance:
        old_theta = theta.copy()
        try:
            inverse_Hessian = np.linalg.inv(hessian_H(theta, x, y))
        except np.linalg.LinAlgError:
            print("Ran into Non-Invertible Hessian in Step {}. Trying Pseudo-Inverse...".format(step))
            inverse_Hessian = np.linalg.pinv(hessian_H(theta, x, y))
        gradient = gradient_L(theta, x, y)
        theta = theta - inverse_Hessian.dot(gradient)
        diff = np.linalg.norm(theta - old_theta, ord=1)
        step = step + 1
        theta_list.append(theta)
    return theta_list

def main():
    # Parse in the raw data from x.txt and y.txt
    # Note to self (and professor/TA) - Never used pandas before, this took *forever* to figure out
    raw_x = pd.read_csv("x.txt", sep="\ +", names=["x_1","x_2"], header=None, engine='python')
    raw_y = pd.read_csv('y.txt', sep='\ +', names=["y"], header=None, engine='python')
    raw_y = raw_y.astype(int)

    # Form the X and Y matrices (X needs a column of 1s to account for theta0)
    x_mat = np.hstack([np.ones((raw_x.shape[0], 1)), raw_x[["x_1","x_2"]].to_numpy()])
    y_mat = raw_y["y"].to_numpy()

    # Initialize theta_0 matrix to a zero matrix as long as the x_mat (including the padded 1s)
    theta_0 = np.zeros(x_mat.shape[1])

    # Set the Tolerance (theta must change by a delta smaller than this to be considered converged)
    # I picked 1e-6 because it's fairly small and I don't want it to run *forever*
    tolerance = 1e-6
    
    # Execute the Newton Raphson Regression Method using the above inputs
    theta = newton_raphson_regression(theta_0, x_mat, y_mat, tolerance)
    
    # Outputting the result
    print("After Convergence, Theta is: ")
    print(theta)

if __name__ == "__main__":
    main()
