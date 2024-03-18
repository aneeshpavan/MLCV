# No additional 3rd party external libraries are allowed
import numpy as np
from numba import cuda

@cuda.jit(device=True)
class BatchNorm:
    def __init__(self, D, eps=1e-5):
        self.D = D
        self.eps = eps
        self.momemtum = 0.9
        self.gamma = np.ones((1, D))
        self.beta = np.zeros((1, D))
        self.running_mean = np.zeros((1, D))
        self.running_variance = np.zeros((1, D))
        self.temp = None

    def batchnorm(self, x, mode='train'):
        if mode == 'train':
            mean = np.mean(x, axis=0, keepdims=True)
            variance = np.var(x, axis=0, keepdims=True)

            self.running_mean = self.momemtum * \
                self.running_mean + (1-self.momemtum) * mean
            self.running_variance = self.momemtum * \
                self.running_variance + (1-self.momemtum) * variance

            x_hat = (x - mean) / np.sqrt(variance + self.eps)

            output = self.gamma * x_hat + self.beta

        elif mode == 'test':
            x_hat = (x - self.running_mean) / \
                np.sqrt(self.running_variance + self.eps)

            output = self.gamma * x_hat + self.beta

        self.temp = (x, x_hat, self.gamma, self.running_mean,
                     self.running_variance)

        return output

    def batchnorm_grad(self, dl_dy):
        x, x_norm, gamma, mean, variance = self.temp

        dl_x_hat = dl_dy * gamma

        dl_var = np.sum(dl_x_hat * (x - mean) * -0.5 *
                        np.power(variance + self.eps, -1.5), axis=0)
        dl_mean = np.sum(dl_x_hat * -1 / np.sqrt(variance + self.eps), axis=0,
                         keepdims=True) + dl_var * np.mean(-2 * (x - mean), axis=0)

        dl_xi = dl_x_hat / np.sqrt(variance + self.eps) + dl_var * \
            2 * (x - mean) / x.shape[0] + dl_mean / x.shape[0]

        dl_gamma = np.sum(dl_dy * x_norm, axis=0)
        dl_beta = np.sum(dl_dy, axis=0)

        return dl_xi, dl_gamma, dl_beta
