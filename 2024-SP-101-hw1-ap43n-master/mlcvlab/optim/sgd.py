# No additional 3rd party external libraries are allowed
import random
import numpy as np

def SGD(model, train_X, train_y, lr=0.1, R=100):
    INPUT_LAYER_NEURONS = train_X.shape[1]

    print('SGD optimizer starts here:')
    for i in range(R):
        lr = lr * 0.5 if i == R // 2 else lr
        print(f"R: {i} started")

        for layer_idx, layer in enumerate(model.layers):
            W = layer.W
            grad_W = np.zeros_like(W)
            
            rand_row = random.choice(range(W.shape[0]))
            rand_col = 0 if layer_idx == 1 else random.choice(range(W.shape[1]))
            
            emp_loss_grad_ = model.emp_loss_grad(train_X, train_y, [model.layers[0].W, model.layers[1].W], -1)
            grad_W[rand_row, rand_col] = emp_loss_grad_[layer_idx][rand_row, rand_col]
            layer.W -= lr * grad_W

        print(f"R: {i} completed")  
    return [model.layers[0].W, model.layers[1].W]