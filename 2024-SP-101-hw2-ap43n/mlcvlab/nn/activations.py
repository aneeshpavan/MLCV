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
    expX = np.exp(x)
    softmax = np.divide(expX, np.transpose(np.tile(np.sum(expX, axis=1), (np.shape(x)[1], 1))))
    return softmax
    # raise NotImplementedError("Softmax function not implemented")

def softmax_grad(z):
    z_shape = np.shape(z)[1]
    softmax_grad = np.multiply(-1 * np.tile(np.transpose(z), (1, z_shape)), np.tile(z, (z_shape, 1)) - np.identity(z_shape))
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
