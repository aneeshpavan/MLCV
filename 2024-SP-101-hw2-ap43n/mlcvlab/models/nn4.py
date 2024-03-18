import numpy as np
from mlcvlab.nn.losses import l2, l2_grad, cross_entropy, cross_entropy_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.batchnorm import BatchNorm


class NN4():
    def __init__(self, use_batchnorm=False, dropout_param=0):
        self.layers = [
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, sigmoid)]

        self.use_batchnorm = use_batchnorm

        self.dropout_param = dropout_param

        self.W = []
        self.bn1 = BatchNorm(500)
        self.bn2 = BatchNorm(100)
        self.bn3 = BatchNorm(50)
        self.y_tilde = None

    def nn4(self, x, mode):
        if self.use_batchnorm:
            # Layer 1
            self.z1, self.mask1 = linear(
                x, self.layers[0].W, self.dropout_param, mode)
            self.z1_tilde = self.layers[0].activation(self.z1)
            self.y1 = self.bn1.batchnorm(self.z1_tilde, mode)

            # Layer 2
            self.z2, self.mask2 = linear(
                self.y1, self.layers[1].W, self.dropout_param, mode)
            self.z2_tilde = self.layers[1].activation(self.z2)
            self.y2 = self.bn2.batchnorm(self.z2_tilde, mode)

            # Layer 3
            self.z3, self.mask3 = linear(
                self.y2, self.layers[2].W, self.dropout_param, mode)
            self.z3_tilde = self.layers[2].activation(self.z3)
            self.y3 = self.bn3.batchnorm(self.z3_tilde, mode)

            # Layer 4
            self.z4, self.mask4 = linear(
                self.y3, self.layers[3].W, self.dropout_param, mode)
            self.y_tilde = self.layers[3].activation(self.z4)

            return self.y_tilde

        else:
            # Layer 1
            self.z1, self.mask1 = linear(
                x, self.layers[0].W, self.dropout_param, mode)
            self.z1_tilde = self.layers[0].activation(self.z1)

            # Layer 2
            self.z2, self.mask2 = linear(
                self.z1_tilde, self.layers[1].W, self.dropout_param, mode)
            self.z2_tilde = self.layers[1].activation(self.z2)

            # Layer 3
            self.z3, self.mask3 = linear(
                self.z2_tilde, self.layers[2].W, self.dropout_param, mode)
            self.z3_tilde = self.layers[2].activation(self.z3)

            # Layer 4
            self.z4, self.mask4 = linear(
                self.z3_tilde, self.layers[3].W, self.dropout_param, mode)
            self.y_tilde = self.layers[3].activation(self.z4)

            return self.y_tilde

    def grad(self, x, y, W, mode):
        self.W = W
        self.layers[0].W = self.W[0]
        self.layers[1].W = self.W[1]
        self.layers[2].W = self.W[2]
        self.layers[3].W = self.W[3]
        if self.use_batchnorm:
            y_tilde = self.nn4(x, mode)

            # Layer 4 back prop
            dl_z4 = l2_grad(y, self.y_tilde) * \
                sigmoid_grad(self.y_tilde)
            dl_w4 = self.y3.T.dot(linear_grad(dl_z4, self.mask4, mode)) / 100

            # Layer 3 back prop
            dl_z3tilde, dl_gamma3, dl_beta3 = self.bn3.batchnorm_grad(
                (dl_z4 * self.mask4).dot(self.layers[3].W.T))

            dl_z3 = dl_z3tilde * relu_grad(self.z3_tilde)
            dl_w3 = self.y2.T.dot(linear_grad(dl_z3, self.mask3, mode)) / 100

            # Layer 2 back prop
            dl_z2tilde, dl_gamma2, dl_beta2 = self.bn2.batchnorm_grad(
                (dl_z3 * self.mask3).dot(self.layers[2].W.T))

            dl_z2 = dl_z2tilde * relu_grad(self.z2_tilde)
            dl_w2 = self.y1.T.dot(linear_grad(dl_z2, self.mask2, mode)) / 100

            # Layer 1 back prop
            dl_z1tilde, dl_gamma1, dl_beta1 = self.bn1.batchnorm_grad(
                (dl_z2 * self.mask2).dot(self.layers[1].W.T))

            dl_z1 = dl_z1tilde * relu_grad(self.z1_tilde)
            dl_w1 = x.T.dot(linear_grad(dl_z1, self.mask1, mode)) / 100

            return [dl_w1, dl_w2, dl_w3, dl_w4, dl_gamma1, dl_beta1, dl_gamma2, dl_beta2, dl_gamma3, dl_beta3]

        else:
            y_tilde = self.nn4(x, mode)

            # Layer 4 back prop
            dl_z4 = l2_grad(y, y_tilde) * sigmoid_grad(self.y_tilde)
            dl_w4 = self.z3_tilde.dot((dl_z4 * self.mask4).T) / 100

            # Layer 3 back prop
            dl_z3 = self.layers[3].W.dot(
                (dl_z4 * self.mask4)) * relu_grad(self.z3_tilde)
            dl_w3 = self.z2_tilde.dot((dl_z3 * self.mask3).T) / 100

            # Layer 2 back prop
            dl_z2 = self.layers[2].W.dot(
                (dl_z3 * self.mask3)) * relu_grad(self.z2_tilde)
            dl_w2 = self.z1_tilde.dot(
                (dl_z2 * self.mask2).T) / 100

            # Layer 1 back prop
            dl_z1 = self.layers[1].W.dot(
                (dl_z2 * self.mask2)) * relu_grad(self.z1_tilde)
            dl_w1 = x.dot((dl_z1 * self.mask1).T) / 100

            return [dl_w1, dl_w2, dl_w3, dl_w4]

    def emp_loss_grad(self, train_X, train_y, W, layer, mode='test'):
        train_X = np.concatenate([train_X, np.ones((100, 1))], axis=1)
        train_y = train_y.reshape((-1, 1))
        return self.grad(train_X, train_y, W, mode)