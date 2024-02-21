class Layer():
    def __init__(self, W, activation, activation_grad):
        self.W  = W
        self.activation = activation
        self.activation_grad = activation_grad