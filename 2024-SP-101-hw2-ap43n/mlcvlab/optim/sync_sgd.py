# No additional 3rd party external libraries are allowed
import numpy as np
from numba import jit, prange
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

@jit(forceobj=True)
def sync_sgd(model, X_train_batches, y_train_batches, lr, R=10, mode='test'):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''
    print('Starting SGD...')
    #TODO
    for iteration in range(R):
        # Iterate through each batch
        for batch_index in range(len(X_train_batches)):
            X_batch = X_train_batches[batch_index]
            y_batch = y_train_batches[batch_index]

            # Append ones for bias
            ones_column = np.ones((X_batch.shape[0], 1))
            X_batch_augmented = np.concatenate([X_batch, ones_column], axis=1)
            y_batch_reshaped = y_batch.reshape((-1, 1))
            
            # Forward pass
            predictions = model.nn4(X_batch_augmented, mode)
            loss = np.mean(np.square(y_batch_reshaped - predictions))
            
            # Compute gradients
            weights = [layer.W for layer in model.layers]
            gradients = model.grad(X_batch_augmented, y_batch_reshaped, weights, mode)
            
            # Update weights and batch normalization parameters
            for layer_index, layer in enumerate(model.layers):
                layer.W -= lr * gradients[layer_index]
            
            if model.use_batchnorm:
                bn_params = gradients[len(model.layers):]
                model.bn1.gamma -= lr * bn_params[0]
                model.bn1.beta -= lr * bn_params[1]
                model.bn2.gamma -= lr * bn_params[2]
                model.bn2.beta -= lr * bn_params[3]
                model.bn3.gamma -= lr * bn_params[4]
                model.bn3.beta -= lr * bn_params[5]
        
        print('Iteration', iteration + 1, 'completed with loss:', "{:.6f}".format(loss))

    updated_weights = [layer.W for layer in model.layers]
    print("Completed training model - final W : {}".format(updated_weights))
    
    return model