import numpy as np

class Dense:

    def __init__(self, input_dim, output_dim, init="xavier"):

        if init == "xavier":
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

        self.input = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_out):

        self.grad_W = self.input.T @ grad_out
        self.grad_b = np.sum(grad_out, axis=0, keepdims=True)

        return grad_out @ self.W.T