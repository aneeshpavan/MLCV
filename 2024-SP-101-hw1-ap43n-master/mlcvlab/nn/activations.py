# No additional 3rd party external libraries are allowed
import numpy as np

def relu(x):
    relu = (x >= 0) * x
    return relu
    # raise NotImplementedError("ReLU function not implemented")

def relu_grad(z):
    relu_grad = (z > 0) * 1
    return relu_grad
    # raise NotImplementedError("Gradient of ReLU function not implemented")

def sigmoid(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid
    # raise NotImplementedError("Sigmoid function not implemented")
    
def sigmoid_grad(z):
    sigmoid_grad = z * (1 - z)
    return sigmoid_grad
    # raise NotImplementedError("Gradient of Sigmoid function not implemented")

def softmax(x):
    if len(x.shape) > 1:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / np.sum(e_x, axis=1, keepdims=True)
    else:
        e_x = np.exp(x - np.max(x))
        softmax = e_x / np.sum(e_x)
    return softmax
    # raise NotImplementedError("Softmax function not implemented")

def softmax_grad(z):
    z = np.array(z).reshape(-1, 1)
    softmax_grad = np.diagflat(z) - np.dot(z, z.T)
    return softmax_grad
    # raise NotImplementedError("Gradient of Softmax function not implemented")

def tanh(x):
    tanh = np.tanh(x)
    return tanh
    # raise NotImplementedError("Tanh function not implemented")

def tanh_grad(z):
    tanh_grad = 1 - np.square(np.tanh(z))
    return tanh_grad
    # raise NotImplementedError("Gradient of Tanh function not implemented")
