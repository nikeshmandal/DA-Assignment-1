import numpy as np

class CrossEntropy:
    def forward(self, logits, y):
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.y = y
        N = y.shape[0]
        return -np.mean(np.log(self.probs[np.arange(N), y]))

    def backward(self):
        N = self.y.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y] -= 1
        return grad / N


class MSE:
    def forward(self, pred, y):
        self.pred = pred
        self.y = y
        return np.mean((pred - y)**2)

    def backward(self):
        return 2 * (self.pred - self.y) / self.y.size