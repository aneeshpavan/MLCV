import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer

class NN2():
    def __init__(self):
        self.layers = [
            Layer(None, relu, relu_grad), Layer(None, sigmoid, sigmoid_grad)]

    def nn2(self, x):
        W1 = self.layers[0].W
        W2 = self.layers[1].W
        y_hat = self.layers[1].activation(
            linear(
                self.layers[0].activation(linear(x, W1)),
                W2
            )
        )
        return y_hat

    def grad(self, x, y, W):
        layer1, layer2 = self.layers[:2]
        W1, W2 = layer1.W, layer2.W
        sigma1, sigma2 = layer1.activation, layer2.activation
        sigma1_grad, sigma2_grad = layer1.activation_grad, layer2.activation_grad

        z1, acti = linear(x, W1), sigma1(linear(x, W1))
        z2, y_hat = linear(acti, W2), sigma2(linear(acti, W2)).T

        loss = l2(y, y_hat)
        print(loss)
        grad_loss = l2_grad(y, y_hat)
        grad_loss_lin2 = np.multiply(grad_loss, sigma2_grad(z2).T)
        grad_loss_lin1 = np.dot(W2, grad_loss_lin2.T) * sigma1_grad(z1)
        grad_loss_weight1 = np.dot(linear_grad(x), grad_loss_lin1.T)
        grad_loss_weight2 = np.dot(sigma1_grad(acti), grad_loss_lin2)
        return [grad_loss_weight1, grad_loss_weight2]

    def emp_loss_grad(self, train_X, train_y, W, layer):
        num = train_X.shape[0]
        inv_num = 1 / num
        new_train_X = np.concatenate((np.transpose(train_X), -1 * np.ones((1, num))))
        train_set = self.grad(new_train_X, train_y, W)
        avg_grad_0 = inv_num * train_set[0]
        avg_grad_1 = inv_num * train_set[1]
        return [avg_grad_0, avg_grad_1]