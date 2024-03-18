# No additional 3rd party external libraries are allowed
import numpy as np
from mlcvlab.nn.dropout import dropout, dropout_grad

def linear(x, W, p=0.5, mode="test"):
    linear = dropout(np.dot(x, W), p, mode)
    return linear

def linear_grad(x, mask, mode="test"):
    linear_grad = dropout_grad(x, mask, mode)
    return linear_grad

def radial(x, W):
    # TODO
    raise NotImplementedError("Radial Basis function not implemented")
    
def radial_grad(loss_grad_y, x, W):
    # TODO
    raise NotImplementedError("Gradient of Radial Basis function not implemented")