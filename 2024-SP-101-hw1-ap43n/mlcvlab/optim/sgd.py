import numpy as np
import random

def SGD(model, train_X, train_y, lr=0.1, R=10):
    print('SGD optimizer starts here:')
    for i in range(R):
        lr = lr * 0.5 if i == R // 2 else lr
        print(f"R: {i} started")

        # Prepare zeroed-out weight matrices for gradient calculation
        grad_W1 = np.zeros_like(model.layers[0].W)
        grad_W2 = np.zeros_like(model.layers[1].W)
        
        # Randomly select indices for the weight to update
        rand_row1 = random.choice(range(grad_W1.shape[0]))
        rand_col1 = random.choice(range(grad_W1.shape[1]))
        rand_row2 = random.choice(range(grad_W2.shape[0]))
        rand_col2 = 0  # Assuming the second layer is a vector or has a single column

        # Calculate gradients with zeroed-out weights, then only update the selected weight
        emp_loss_grad_ = model.emp_loss_grad(train_X, train_y, [grad_W1, grad_W2], -1)
        
        # Apply the calculated gradient to the randomly selected weight
        grad_W1[rand_row1, rand_col1] = emp_loss_grad_[0][rand_row1, rand_col1]
        model.layers[0].W[rand_row1, rand_col1] -= lr * grad_W1[rand_row1, rand_col1]

        grad_W2[rand_row2, rand_col2] = emp_loss_grad_[1][rand_row2, rand_col2]
        model.layers[1].W[rand_row2, rand_col2] -= lr * grad_W2[rand_row2, rand_col2]

        print(f"R: {i} completed")  
    return [model.layers[0].W, model.layers[1].W]
