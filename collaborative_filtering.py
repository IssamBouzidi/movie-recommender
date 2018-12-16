#!/usr/bin/env python3

"""
Author: Suraj Regmi
Date: 16th December, 2018
Description: Implementation of collaborative filtering from scratch using NumPy
Application Case: Recommending movies to the users based on their ratings on other movies i.e user-based recommendation
system
"""

# Check if the NumPy module is available.
try:
    import numpy as np
except ImportError:
    print("This implementation requires the NumPy module.\nPlease install NumPy by:\npip(or pip3) install NumPy")
    exit(0)

"""
Symbols and Definition:
n_u : No of users
n_m : No of movies

@INPUT:
    R     : Ratings matrix having dimension (n_u x n_m)
    K     : No of latent features to be used
    U     : User features matrix (n_u x K)
    V     : Movie features matrix (n_m x K)
    steps : the maximum number of iterations to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
    
@OUTPUT:
    Rating matrix with all the predicted ratings.
"""
def collaborative_filtering(R, U, V, K, steps=5000, alpha=0.01, beta=0.02):

    n_u = U.shape[0]
    n_m = V.shape[0]

    V = V.T

    # Run steps number of iteration training
    for _ in range(steps):
        for i in range(n_u):
            for j in range(n_m):
                if R[i, j] != 0:
                    for k in range(K):

                        # Update weights according to gradient descent algorithm
                        U[i, k] = U[i, k] - alpha * (2 * (np.dot(U[i, :], V[:, j]) - R[i, j]) * V[k, j] + beta * 2 * U[i, k])
                        V[k, j] = V[k, j] - alpha * (2 * (np.dot(U[i, :], V[:, j]) - R[i, j]) * U[i, k] + beta * 2 * V[k, j])

    # Return the full ratings matrix
    return np.matmul(U, V)


if __name__ == "__main__":

    # Take a sample rating matrix
    R = [
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4],
    ]

    R = np.array(R)

    n_u = len(R)
    n_m = len(R[0])
    K = 2

    U = np.random.rand(n_u, K)
    V = np.random.rand(n_m, K)

    # Take 2 latent features and train
    R = collaborative_filtering(R, U, V, K)

    # Print full rating matrix
    print(R)
