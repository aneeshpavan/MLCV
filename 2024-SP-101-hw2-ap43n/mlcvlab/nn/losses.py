# No additional 3rd party external libraries are allowed
import numpy as np

def l2(y, y_hat):
    l2 = np.mean(np.square(y - y_hat))
    return l2
    # raise NotImplementedError("l2 loss function not implemented")

def l2_grad(y, y_hat):
    l2_grad = np.divide(2 * (y_hat - y), len(y))
    return l2_grad
    # raise NotImplementedError("Gradiant of l2 loss function not implemented")

def cross_entropy(y, y_hat):
    EPSILON = 1e-8
    y_hat_clipped = np.clip(y_hat, EPSILON, 1 - EPSILON)
    cross_entropy = -np.mean(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
    return cross_entropy
    # raise NotImplementedError("Cross entropy loss function not implemented")
    
def cross_entropy_grad(y, y_hat):
    N = y.shape[0]
    EPSILON = 1e-8
    y_hat_clipped = np.clip(y_hat, EPSILON, 1 - EPSILON)
    cross_entropy_grad = -(y / y_hat_clipped - (1 - y) / (1 - y_hat_clipped)) / N    
    return cross_entropy_grad
    # raise NotImplementedError("Gradiant of Cross entropy loss function not implemented")