# No additional 3rd party external libraries are allowed
import numpy as np
import random
from numba import cuda, njit,prange

def sync_sgd_gpu(model, X_train_batches, y_train_batches, lr, R=10, mode='test'):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''
    #TODO
    original_weights = [layer.W.copy() for layer in model.layers]
    
    X_batches_gpu = cuda.to_device(X_train_batches)
    y_batches_gpu = cuda.to_device(y_train_batches)
    model_gpu = cuda.to_device(model)
    
    EPSILON = 1e-8
    
    num_blocks = np.shape(X_train_batches)[2]
    threads_per_block = min(1024, 60000 // num_blocks)
    
    print('Starting SGD...')
    #TODO
    for Iteration in range(R):
        param_list = [layer.W for layer in model.layers]
        if model.use_batchnorm:
            param_list += [bn_param for bn_layer in [model.bn1, model.bn2, model.bn3] for bn_param in (bn_layer.gamma, bn_layer.beta)]
        
        params_gpu = cuda.to_device(param_list)
        
        model_gpu.emp_loss_grad[num_blocks, threads_per_block](X_batches_gpu, y_batches_gpu, mode, params_gpu)
        
        cuda.synchronize()
        
        updated_params = params_gpu.copy_to_host()
        update_idx = 0
        for layer in model.layers:
            layer.W -= lr * updated_params[update_idx]
            update_idx += 1
        if model.use_batchnorm:
            for bn_layer in [model.bn1, model.bn2, model.bn3]:
                bn_layer.gamma -= lr * updated_params[update_idx]
                bn_layer.beta -= lr * updated_params[update_idx + 1]
                update_idx += 2
                
    final_weights = [layer.W for layer in model.layers]
    print(f"Completed training model - final W : {final_weights}")
    return model


def sync_sgd(model, X_train_batches, y_train_batches, lr, R=10, mode='test'):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''
    print('Starting SGD...')
    #TODO
    for Iteration in range(R):
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
            gradients = model.grad(X_batch_augmented, y_batch_reshaped, [layer.W for layer in model.layers], mode)
            
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
        
        print(f'Iteration {Iteration + 1} completed with loss: {loss:.6f}')

    updated_weights = [layer.W for layer in model.layers]
    print(f"Completed training model - final W : {updated_weights}")
    
    return model