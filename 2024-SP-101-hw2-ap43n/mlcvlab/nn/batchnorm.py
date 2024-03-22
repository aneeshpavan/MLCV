# No additional 3rd party external libraries are allowed
import numpy as np

class BatchNorm:
    def __init__(self, DIMENSION, EPSILON=1e-5):
        self.DIMENSION = DIMENSION
        self.EPSILON = EPSILON
        self.gamma = np.ones((1, DIMENSION))
        self.beta = np.zeros((1, DIMENSION))
        self.moving_mean = np.zeros((1, DIMENSION))
        self.moving_variance = np.zeros((1, DIMENSION))

    def batchnorm(self, x, mode='train'):
        if mode == 'train':
            mean = np.mean(x, axis=0, keepdims=True)
            variance = np.var(x, axis=0, keepdims=True)
            x_norm = (x - mean) / np.sqrt(variance + self.EPSILON)
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_variance = 0.9 * self.moving_variance + 0.1 * variance
            out = self.gamma * x_norm + self.beta

        elif mode == 'test':
            x_norm = (x - self.moving_mean) / np.sqrt(self.moving_variance + self.EPSILON)
            out = self.gamma * x_norm + self.beta

        self.cache = (x, x_norm, self.gamma, mean if mode == 'train' else self.moving_mean, variance if mode == 'train' else self.moving_variance)

        return out
    # raise NotImplementedError("Batchnorm Not Implemented")

    def batchnorm_grad(self, del_l_y):
        x, x_norm, gamma, mean, variance = self.cache

        del_l_x_norm = del_l_y * gamma

        del_var = np.sum(del_l_x_norm * (x - mean) * -0.5 * np.power(variance + self.EPSILON, -1.5), axis=0, keepdims=True)
        del_mean = np.sum(del_l_x_norm * -1 / np.sqrt(variance + self.EPSILON), axis=0, keepdims=True) + del_var * np.mean(-2 * (x - mean), axis=0, keepdims=True)

        del_l_x = del_l_x_norm / np.sqrt(variance + self.EPSILON) + del_var * 2 * (x - mean) / x.shape[0] + del_mean / x.shape[0]
        
        del_l_gamma = np.sum(del_l_y * x_norm, axis=0, keepdims=True)
        del_l_beta = np.sum(del_l_y, axis=0, keepdims=True)

        return del_l_x, del_l_gamma, del_l_beta
    # raise NotImplementedError("Gradiant of Batchnorm Not Implemented")
