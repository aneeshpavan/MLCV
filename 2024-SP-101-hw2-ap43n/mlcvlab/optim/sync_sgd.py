# No additional 3rd party external libraries are allowed
import numpy as np

def sync_sgd(model, X_train_batches, y_train_batches, lr, R=10, mode='test'):
    for i in range(R):
        old_w1 = model.layers[0].W
        old_w2 = model.layers[1].W
        old_w3 = model.layers[2].W
        old_w4 = model.layers[3].W

        print(f"iteration: {i}")

        for batch in range(len(X_train_batches)):
            X_train = X_train_batches[batch]
            y_train = y_train_batches[batch]
            Dw = np.ones((100, 1))

            X_train = np.concatenate([X_train, Dw], axis=1)
            y_train = y_train.reshape((-1, 1))

            model.nn4(X_train, mode)
            emp_loss = model.grad(X_train, y_train, [
                                 old_w1, old_w2, old_w3, old_w4], mode)

            model.layers[0].W = old_w1 - (lr * (emp_loss[0]))
            model.layers[1].W = old_w2 - (lr * (emp_loss[1]))
            model.layers[2].W = old_w3 - (lr * (emp_loss[2]))
            model.layers[3].W = old_w4 - (lr * (emp_loss[3]))

            if model.use_batchnorm:
                model.bn1.gamma = model.bn1.gamma - (lr * (emp_loss[4]))
                model.bn1.beta = model.bn1.beta - (lr * (emp_loss[5]))
                model.bn2.gamma = model.bn2.gamma - (lr * (emp_loss[6]))
                model.bn2.beta = model.bn2.beta - (lr * (emp_loss[7]))
                model.bn3.gamma = model.bn3.gamma - (lr * (emp_loss[8]))
                model.bn3.beta = model.bn3.beta - (lr * (emp_loss[9]))
    return model