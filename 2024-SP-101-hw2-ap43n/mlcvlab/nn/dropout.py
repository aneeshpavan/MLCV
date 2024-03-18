# No additional 3rd party external libraries are allowed
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def dropout(x, p=0.05, mode='test'):
    if mode == 'train':
        mask = np.random.binomial(1, p, size=x.shape) * (1.0 / p)
        return x * mask, mask

    else:
        return x, None

@cuda.jit(device=True)
def dropout_grad(z, mask, mode='test'):
    return z * mask
