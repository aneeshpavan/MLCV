import numpy as np

def Adam(model, train_X, train_y, hidden_layer_neurons, R=10):
    ALPHA = 1e-2
    BETA_1 = 0.9
    BETA_2 = 0.999
    EPSILON = 1e-8
    # These variables are now explicitly acknowledged but will be used as before
    INPUT_LAYER_NEURONS = train_X.shape[1]
    HIDDEN_LAYER_NEURONS = hidden_layer_neurons

    # Initialize moment vectors based on the shapes of the weight matrices
    m1 = np.zeros_like(model.layers[0].W)
    s1 = np.zeros_like(model.layers[0].W)
    m2 = np.zeros_like(model.layers[1].W)
    s2 = np.zeros_like(model.layers[1].W)

    print('ADAM optimizer starts here:')
    for i in range(1, R+1):
        print(f"R: {i} started")

        # Calculate gradients
        emp_loss_grad = model.emp_loss_grad(train_X, train_y, [model.layers[0].W, model.layers[1].W], -1)

        # Update moment estimates
        m1 = BETA_1 * m1 + (1 - BETA_1) * emp_loss_grad[0]
        s1 = BETA_2 * s1 + (1 - BETA_2) * np.square(emp_loss_grad[0])
        m2 = BETA_1 * m2 + (1 - BETA_1) * emp_loss_grad[1]
        s2 = BETA_2 * s2 + (1 - BETA_2) * np.square(emp_loss_grad[1])

        # Apply bias correction
        m1_hat = m1 / (1 - BETA_1**i)
        s1_hat = s1 / (1 - BETA_2**i)
        m2_hat = m2 / (1 - BETA_1**i)
        s2_hat = s2 / (1 - BETA_2**i)

        # Update weights
        model.layers[0].W -= ALPHA * m1_hat / (np.sqrt(s1_hat) + EPSILON)
        model.layers[1].W -= ALPHA * m2_hat / (np.sqrt(s2_hat) + EPSILON)

        print(f"R: {i} completed")

    return [model.layers[0].W, model.layers[1].W]
