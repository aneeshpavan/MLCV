import numpy as np

def Adam(model, train_X, train_y,hidden_layer_neurons, R=100):
    ALPHA = 1e-2
    BETA_1 = 0.9
    BETA_2 = 0.999
    EPSILON = 1e-8
    INPUT_LAYER_NEURONS = train_X.shape[1]
    HIDDEN_LAYER_NEURONS = hidden_layer_neurons

    m1 = np.zeros((INPUT_LAYER_NEURONS + 1, HIDDEN_LAYER_NEURONS + 1))
    s1 = np.zeros((INPUT_LAYER_NEURONS + 1, HIDDEN_LAYER_NEURONS + 1))

    m2 = np.zeros((HIDDEN_LAYER_NEURONS + 1, 1))
    s2 = np.zeros((HIDDEN_LAYER_NEURONS + 1, 1))
    
    print('ADAM optimizer starts here:')
    for i in range(R):
        print(f"R: {i} started")
        old_W1 = model.layers[0].W
        old_W2 = model.layers[1].W
        new_W = [old_W1, old_W2]

        emp_loss_grad_no_momentum = model.emp_loss_grad(train_X, train_y, new_W, -1)
        model.layers[0].W += BETA_1 * m1
        model.layers[1].W += BETA_1 * m2
        emp_loss_grad_with_momentum = model.emp_loss_grad(train_X, train_y, new_W, -1)

        new_M1 = BETA_1 * m1 + (1 - BETA_1) * emp_loss_grad_with_momentum[0]
        new_S1 = BETA_2 * s1 + (1 - BETA_2) * np.power(emp_loss_grad_no_momentum[0], 2)

        new_M2 = BETA_1 * m2 + (1 - BETA_1) * emp_loss_grad_with_momentum[1]
        new_S2 = BETA_2 * s2 + (1 - BETA_2) * np.power(emp_loss_grad_no_momentum[1], 2)

        bias_corrected_M1 = new_M1 / (1 - BETA_1**(i+1))
        bias_corrected_M2 = new_M2 / (1 - BETA_1**(i+1))

        bias_corrected_S1 = new_S1 / (1 - BETA_2**(i+1))
        bias_corrected_S2 = new_S2 / (1 - BETA_2**(i+1))

        model.layers[0].W = old_W1 - ((ALPHA / (np.sqrt(bias_corrected_S1) + EPSILON)) * bias_corrected_M1)
        model.layers[1].W = old_W2 - ((ALPHA / (np.sqrt(bias_corrected_S2) + EPSILON)) * bias_corrected_M2)

        m1 = new_M1
        m2 = new_M2
        s1 = new_S1
        s2 = new_S2
        print(f"R: {i} completed")

    return [model.layers[0].W, model.layers[1].W]