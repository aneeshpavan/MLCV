import numpy as np
from mlcvlab.nn.losses import l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.batchnorm import BatchNorm

class NN4():
    def __init__(self, use_batchnorm=False, dropout_param=0.25):
        self.layers = [Layer(None, relu), Layer(None, relu), Layer(None, relu), Layer(None, sigmoid)]
        self.use_batchnorm = use_batchnorm
        #used in dropout implementation
        self.dropout_param = dropout_param
        
        if self.use_batchnorm:
            self.bn1 = BatchNorm(500)
            self.bn2 = BatchNorm(100)
            self.bn3 = BatchNorm(50)

    def nn4(self, x, mode):
        # TODO
        if self.use_batchnorm:
            # Layer 1
            self.z1, self.mask1 = linear(x, self.layers[0].W, self.dropout_param, mode)  # Linear transformation with integrated dropout
            self.z1_hat = relu(self.z1)  # ReLU activation function
            self.y1 = self.bn1.batchnorm(self.z1_hat, mode)  # batch normalization

            # Layer 2
            self.z2, self.mask2 = linear(self.y1, self.layers[1].W, self.dropout_param, mode)  # Linear transformation with integrated dropout
            self.z2_hat = relu(self.z2)
            self.y2 = self.bn2.batchnorm(self.z2_hat, mode)

            # Layer 3
            self.z3, self.mask3 = linear(self.y2, self.layers[2].W, self.dropout_param, mode)  # Linear transformation with integrated dropout
            self.z3_hat = relu(self.z3)
            self.y3 = self.bn3.batchnorm(self.z3_hat, mode)

            # Layer 4
            self.z4, self.mask4 = linear(self.y3, self.layers[3].W, self.dropout_param, mode)  # Linear transformation with integrated dropout
            self.y_hat = sigmoid(self.z4)

            return self.y_hat
        # raise NotImplementedError("NN4 Batchnorm model not implemented")

        else:
            # Layer 1
            self.z1, self.mask1 = linear(x, self.layers[0].W, self.dropout_param, mode)
            self.z1_hat = self.layers[0].activation(self.z1)
            
            # Layer 2
            self.z2, self.mask2 = linear(self.z1_hat, self.layers[1].W, self.dropout_param, mode)
            self.z2_hat = self.layers[1].activation(self.z2)
            
            # Layer 3
            self.z3, self.mask3 = linear(self.z2_hat, self.layers[2].W, self.dropout_param, mode)
            self.z3_hat = self.layers[2].activation(self.z3)
            
            # Layer 4
            self.z4, self.mask4 = linear(self.z3_hat, self.layers[3].W, self.dropout_param, mode)
            self.y_hat = self.layers[3].activation(self.z4)
            
            return self.y_hat
        # raise NotImplementedError("NN4 Without Batchnorm model not implemented")
        

    def grad(self, x, y, W, mode):
        self.W = W
        self.layers[0].W = self.W[0]
        self.layers[1].W = self.W[1]
        self.layers[2].W = self.W[2]
        self.layers[3].W = self.W[3]
        # TODO  
        if self.use_batchnorm:
            #Layer 4
            loss_grad_yhat = l2_grad(y, self.y_hat)
            del_sigmoid_z4 = sigmoid_grad(self.y_hat)
            
            loss_gradient_z4 = loss_grad_yhat * del_sigmoid_z4
            linear_gradient_w4 = linear_grad(loss_gradient_z4, self.mask4, mode)
            loss_gradient_w4 = self.y3.T.dot(linear_gradient_w4) / 100 
            
            #Layer 3
            linear_gradient_y3_from_z4 = self.layers[3].W.T 
            loss_gradient_y3 =  (loss_gradient_z4 * self.mask4).dot(linear_gradient_y3_from_z4) 
            loss_gradient_z3hat, del_batchnorm_gamma3, del_batchnorm_beta3 = self.bn3.batchnorm_grad(loss_gradient_y3)
            relu_gradient_z3hat = relu_grad(self.z3_hat)
            
            loss_gradient_z3 = loss_gradient_z3hat * relu_gradient_z3hat
            linear_gradient_w3 = linear_grad(loss_gradient_z3, self.mask3, mode)
            loss_gradient_w3 = self.y2.T.dot(linear_gradient_w3) / 100 
            
            #Layer 2
            linear_gradient_y2_from_z3 = self.layers[2].W.T
            loss_gradient_y2 = (loss_gradient_z3 * self.mask3).dot(linear_gradient_y2_from_z3) 
            loss_gradient_z2hat, del_batchnorm_gamma2, del_batchnorm_beta2 = self.bn2.batchnorm_grad(loss_gradient_y2) 
            relu_gradient_z2hat = relu_grad(self.z2_hat) 
            
            loss_gradient_z2 = loss_gradient_z2hat * relu_gradient_z2hat 
            del_z2_w2 = linear_grad(loss_gradient_z2, self.mask2, mode)
            grad_w2 = self.y1.T.dot(del_z2_w2) / 100 
            
            #Layer 1
            del_z2_y1 = self.layers[1].W.T 
            loss_gradient_y1 = (loss_gradient_z2 * self.mask2).dot(del_z2_y1) 
            loss_gradient_z1hat, grad_gamma1, grad_beta1 = self.bn1.batchnorm_grad(loss_gradient_y1)
            del_z1hat_z1 = relu_grad(self.z1_hat) 
            
            loss_gradient_z1 = loss_gradient_z1hat * del_z1hat_z1 
            del_z1_w1 = linear_grad(loss_gradient_z1, self.mask1, mode)
            grad_w1 = x.T.dot(del_z1_w1) / 100 
            
            return [grad_w1, grad_w2, loss_gradient_w3, loss_gradient_w4, grad_gamma1, grad_beta1, del_batchnorm_gamma2, del_batchnorm_beta2, del_batchnorm_gamma3, del_batchnorm_beta3]
        # raise NotImplementedError("NN4 gradient (backpropagation) Batchnorm model not implemented")

        else:
            #Layer 4
            loss_grad_yhat = l2_grad(y, self.y_hat)
            del_sigmoid_z4 = sigmoid_grad(self.y_hat)
            
            loss_gradient_z4 = loss_grad_yhat * del_sigmoid_z4
            linear_gradient_w4 = linear_grad(loss_gradient_z4, self.mask4, mode)
            loss_gradient_w4 = self.y3.T.dot(linear_gradient_w4) / 100 
            
            #Layer 3
            linear_gradient_y3_from_z4 = self.layers[3].W.T 
            loss_gradient_y3 = loss_gradient_z4.dot(linear_gradient_y3_from_z4) * relu_grad(self.z3_hat)  # Assuming ReLU is used in layer 3
            
            linear_gradient_w3 = linear_grad(loss_gradient_y3, self.mask3, mode)
            loss_gradient_w3 = self.y2.T.dot(linear_gradient_w3) / 100 
            
            #Layer 2
            linear_gradient_y2_from_z3 = self.layers[2].W.T
            loss_gradient_y2 = loss_gradient_y3.dot(linear_gradient_y2_from_z3) * relu_grad(self.z2_hat)  # Assuming ReLU is used in layer 2
            
            del_z2_w2 = linear_grad(loss_gradient_y2, self.mask2, mode)
            grad_w2 = self.y1.T.dot(del_z2_w2) / 100 
            
            #Layer 1
            del_z2_y1 = self.layers[1].W.T 
            loss_gradient_y1 = loss_gradient_y2.dot(del_z2_y1) * relu_grad(self.z1_hat)  # Assuming ReLU is used in layer 1
            
            del_z1_w1 = linear_grad(loss_gradient_y1, self.mask1, mode)
            grad_w1 = x.T.dot(del_z1_w1) / 100 
            
            return [grad_w1, grad_w2, loss_gradient_w3, loss_gradient_w4]
        # raise NotImplementedError("NN4 gradient (backpropagation) Without Batchnorm model not implemented")     

    def emp_loss_grad(self, train_X, train_y, W, mode='test'):
        # emp_loss_ = 0
        # emp_loss_grad_ = None
        # TODO
        bias_column = np.ones((train_X.shape[0], 1))
        augmented_trainX = np.concatenate([train_X, bias_column], axis=1)
        reshaped_trainY = train_y.reshape((-1, 1))
        
        return self.grad(augmented_trainX, reshaped_trainY, W, mode)
    # raise NotImplementedError("NN4 Emperical Loss grad not implemented")