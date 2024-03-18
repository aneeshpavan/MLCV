# No additional 3rd party external libraries are allowed
import numpy as np
from numpy.linalg import norm 

def l2(y, y_hat):
    l2 = np.linalg.norm(y - y_hat, ord=2)
    return l2
    # raise NotImplementedError("l2 loss function not implemented")

def l2_grad(y, y_hat, epsilon=1e-9):
    l2_grad = np.divide((y - y_hat), (np.linalg.norm(y - y_hat, ord=2) + epsilon))
    return l2_grad
    # raise NotImplementedError("Gradiant of l2 loss function not implemented")

def cross_entropy(A, Y):
    cross_entropy = np.multiply(-A, np.log(Y)) - np.multiply((1 - A), np.log(1 - Y))
    return cross_entropy
    # raise NotImplementedError("Cross entropy loss function not implemented")
    
def cross_entropy_grad(y, y_hat):
    cross_entropy_grad = np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)
    return cross_entropy_grad
    # raise NotImplementedError("Gradiant of Cross entropy loss function not implemented")