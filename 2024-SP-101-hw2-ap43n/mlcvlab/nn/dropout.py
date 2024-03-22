# No additional 3rd party external libraries are allowed
import numpy as np

def dropout(x, p, mode='test'):
    '''
    Output : should return a tuple containing 
     - z : output of the dropout
     - p : Dropout param
     - mode : 'test' or 'train'
     - mask : 
      - in train mode, it is the dropout mask
      - in test mode, mask will be None.
    
    sample output: (z=, p=0.5, mode='test',mask=None)
    '''
    # TODO Implement logic for both 'test' and 'train' modes.
    if mode == 'train':
        mask = np.random.binomial(1, p, size=x.shape)
        z = (x * mask)/p
    # raise NotImplementedError("Dropout - Test Not Implemented")

    elif mode == 'test':
        mask = None
        z = x
    return z, mask
    # raise NotImplementedError("Dropout - Train Not Implemented")

def dropout_grad(z, mask, mode='test'):

    # TODO Implement the gradient computation for dropout. Note that this is just a constant multiplier since there are no model parameters in a dropout mask. 
    if mode == 'train':
        dropout_grad = z * mask
        return dropout_grad
    # raise NotImplementedError("Gradiant of Dropout - Train Not Implemented")
    
    else:
        return z
    # raise NotImplementedError("Gradiant of Dropout - Test Not Implemented")