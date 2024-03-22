import numpy as np
from mlcvlab.nn.losses import l2, l2_grad, cross_entropy, cross_entropy_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.batchnorm import BatchNorm
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8

class NN4_GPU():
    def __init__(self, use_batchnorm=False, dropout_param=0):
        self.layers = [
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, sigmoid)]
        
        self.use_batchnorm = use_batchnorm
        self.dropout_param = dropout_param
        
        if self.use_batchnorm:
            self.bn1 = BatchNorm(500)
            self.bn2 = BatchNorm(100)
            self.bn3 = BatchNorm(50)

    @cuda.jit(device=True)
    def nn4(self, x, mode):
        # TODO
        if self.use_batchnorm:
            EPSILON=1e-8

            #Layer 1
            self.z1, self.mask1 = linear(x, self.layers[0].W, self.dropout_param, mode)
            self.z1_hat = self.layers[0].activation(self.z1)
            self.y1 = self.bn1.batchnorm(self.z1_hat, mode)
            
            #Layer 2
            self.z2, self.mask2 = linear(self.y1, self.layers[1].W, self.dropout_param, mode)
            self.z2_hat = self.layers[1].activation(self.z2)
            self.y2 = self.bn2.batchnorm(self.z2_hat, mode)
            
            #Layer 3
            self.z3, self.mask3 = linear(self.y2, self.layers[2].W, self.dropout_param, mode)
            self.z3_hat = self.layers[2].activation(self.z3)
            self.y3 = self.bn3.batchnorm(self.z3_hat, mode)
            
            #Layer 4
            self.z4, self.mask4 = linear(self.y3, self.layers[3].W, self.dropout_param, mode)
            self.y_hat = self.layers[3].activation(self.z4)
            
            return self.y_hat
        # raise NotImplementedError("NN4 Batchnorm model not implemented")

        else:
            #Layer 1
            self.z1, self.mask1 = linear(x, self.layers[0].W, self.dropout_param, mode)
            self.z1_hat = self.layers[0].activation(self.z1)
            
            #Layer 2
            self.z2, self.mask2 = linear(self.z1_hat, self.layers[1].W, self.dropout_param, mode)
            self.z2_hat = self.layers[1].activation(self.z2)
            
            #Layer 3
            self.z3, self.mask3 = linear(self.z2_hat, self.layers[2].W, self.dropout_param, mode)
            self.z3_hat = self.layers[2].activation(self.z3)
            
            #Layer 4
            self.z4, self.mask4 = linear(self.z3_hat, self.layers[3].W, self.dropout_param, mode)
            self.y_hat = self.layers[3].activation(self.z4)
            
            return self.y_hat
        # raise NotImplementedError("NN4 Without Batchnorm model not implemented")

    @cuda.jit(device=True)
    def grad(self, x, y, mode):
        
        # TODO  
        if self.use_batchnorm:
            #Layer 4
            del_l_yhat = l2_grad(y, self.y_hat)
            del_yhat_z4 = sigmoid_grad(self.y_hat)
            
            del_l_z4 = del_l_yhat * del_yhat_z4 
            del_z4_w4 = linear_grad(del_l_z4, self.mask4, mode)
            del_l_w4 = self.y3.T.dot(del_z4_w4) / 100 
            
            #Layer 3
            del_z4_y3 = self.layers[3].W.T 
            del_l_y3 =  (del_l_z4 * self.mask4).dot(del_z4_y3)
            del_l_z3hat, del_l_gamma3, del_l_beta3 = self.bn3.batchnorm_grad(del_l_y3)
            del_z3hat_z3 = relu_grad(self.z3_hat)
            
            del_l_z3 = del_l_z3hat * del_z3hat_z3
            del_z3_w3 = linear_grad(del_l_z3, self.mask3, mode)
            del_l_w3 = self.y2.T.dot(del_z3_w3) / 100
            
            #Layer 2
            del_z3_y2 = self.layers[2].W.T
            del_l_y2 = (del_l_z3 * self.mask3).dot(del_z3_y2)
            del_l_z2hat, del_l_gamma2, del_l_beta2 = self.bn2.batchnorm_grad(del_l_y2)
            del_z2hat_z2 = relu_grad(self.z2_hat)
            
            del_l_z2 = del_l_z2hat * del_z2hat_z2
            del_z2_w2 = linear_grad(del_l_z2, self.mask2, mode)
            del_l_w2 = self.y1.T.dot(del_z2_w2) / 100 
            
            #Layer 1 
            del_z2_y1 = self.layers[1].W.T
            del_l_y1 = (del_l_z2 * self.mask2).dot(del_z2_y1) 
            del_l_z1hat, del_l_gamma1, del_l_beta1 = self.bn1.batchnorm_grad(del_l_y1)
            del_z1hat_z1 = relu_grad(self.z1_hat)
            
            del_l_z1 = del_l_z1hat * del_z1hat_z1
            del_z1_w1 = linear_grad(del_l_z1, self.mask1, mode)
            del_l_w1 = x.T.dot(del_z1_w1) / 100
            
            return [del_l_w1, del_l_w2, del_l_w3, del_l_w4, del_l_gamma1, del_l_beta1, del_l_gamma2, del_l_beta2, del_l_gamma3, del_l_beta3]

        else:
            y_hat = self.nn4(x, mode)
            
            #Layer 4
            del_l_yhat = l2_grad(y, y_hat)
            del_yhat_z4 = sigmoid_grad(self.y_hat)
            
            del_l_z4 = del_l_yhat * del_yhat_z4 
            del_l_w4 = self.z3_hat.dot((del_l_z4 * self.mask4).T) / 100
            
            #Layer 3
            del_z4_z3hat = self.layers[3].W 
            del_z3hat_z3 = relu_grad(self.z3_hat) 
            del_l_z3hat = del_z4_z3hat.dot((del_l_z4 * self.mask4)) 
            
            del_l_z3 = del_l_z3hat * del_z3hat_z3 
            del_l_w3 = self.z2_hat.dot((del_l_z3 * self.mask3).T) / 100 
            
            #Layer 2
            del_z3_z2hat = self.layers[2].W 
            del_z2hat_z2 = relu_grad(self.z2_hat) 
            del_l_z2hat = del_z3_z2hat.dot((del_l_z3 * self.mask3)) 
            
            del_l_z2 = del_l_z2hat * del_z2hat_z2 
            del_l_w2 = self.z1_hat.dot((del_l_z2 * self.mask2).T) / 100 
            
            #Layer 1
            del_z2_z1hat = self.layers[1].W 
            del_z1hat_z1 = relu_grad(self.z1_hat) 
            del_l_z1hat = del_z2_z1hat.dot((del_l_z2 * self.mask2))
            
            del_l_z1 = del_l_z1hat * del_z1hat_z1 
            del_l_w1 = x.dot((del_l_z1 * self.mask1).T) / 100 
            
            return [del_l_w1, del_l_w2, del_l_w3, del_l_w4]    

    @cuda.jit(device=True)
    def emp_loss_grad(self, train_X, train_y, mode, emp_loss_gpu):
        bias_column = np.ones((train_X.shape[0], 1))
        augmented_trainX = np.concatenate([train_X, bias_column], axis=1)
        reshaped_trainY = train_y.reshape((-1, 1))
        emp_loss_gpu =  self.grad(train_X, train_y, mode)