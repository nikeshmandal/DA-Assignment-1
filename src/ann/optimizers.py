import numpy as np


class SGD:
    def step(self, layers, lr, wd=0.0):
        for l in layers:
            if hasattr(l, "W"):
                l.W -= lr * (l.grad_W + wd * l.W)
                l.b -= lr * l.grad_b


class Momentum:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.vW = None
        self.vb = None

    def _init(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers if hasattr(l, "W")]
        self.vb = [np.zeros_like(l.b) for l in layers if hasattr(l, "W")]

    def step(self, layers, lr, wd=0.0):
        if self.vW is None:
            self._init(layers)
        i = 0
        for l in layers:
            if hasattr(l, "W"):
                self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * l.grad_W
                self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * l.grad_b
                l.W -= lr * self.vW[i] + wd * lr * l.W
                l.b -= lr * self.vb[i]
                i += 1


class NAG:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.vW = None
        self.vb = None

    def _init(self, layers):
        self.vW = [np.zeros_like(l.W) for l in layers if hasattr(l, "W")]
        self.vb = [np.zeros_like(l.b) for l in layers if hasattr(l, "W")]

    def step(self, layers, lr, wd=0.0):
        if self.vW is None:
            self._init(layers)
        i = 0
        for l in layers:
            if hasattr(l, "W"):
                prev_vW = self.vW[i].copy()
                prev_vb = self.vb[i].copy()
                self.vW[i] = self.beta * self.vW[i] + lr * l.grad_W
                self.vb[i] = self.beta * self.vb[i] + lr * l.grad_b
                l.W -= self.beta * prev_vW + (1 + self.beta) * self.vW[i] + wd * lr * l.W
                l.b -= self.beta * prev_vb + (1 + self.beta) * self.vb[i]
                i += 1


class RMSProp:
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.sW = None
        self.sb = None

    def _init(self, layers):
        self.sW = [np.zeros_like(l.W) for l in layers if hasattr(l, "W")]
        self.sb = [np.zeros_like(l.b) for l in layers if hasattr(l, "W")]

    def step(self, layers, lr, wd=0.0):
        if self.sW is None:
            self._init(layers)
        i = 0
        for l in layers:
            if hasattr(l, "W"):
                self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (l.grad_W ** 2)
                self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (l.grad_b ** 2)
                l.W -= lr * l.grad_W / (np.sqrt(self.sW[i]) + self.eps) + wd * lr * l.W
                l.b -= lr * l.grad_b / (np.sqrt(self.sb[i]) + self.eps)
                i += 1


class Adam:
    def __init__(self, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.mW = None
        self.vW = None
        self.mb = None
        self.vb = None

    def _init(self, layers):
        self.mW = [np.zeros_like(l.W) for l in layers if hasattr(l, "W")]
        self.vW = [np.zeros_like(l.W) for l in layers if hasattr(l, "W")]
        self.mb = [np.zeros_like(l.b) for l in layers if hasattr(l, "W")]
        self.vb = [np.zeros_like(l.b) for l in layers if hasattr(l, "W")]

    def step(self, layers, lr, wd=0.0):
        if self.mW is None:
            self._init(layers)
        self.t += 1
        i = 0
        for l in layers:
            if hasattr(l, "W"):
                gW = l.grad_W + wd * l.W
                gb = l.grad_b
                self.mW[i] = self.b1 * self.mW[i] + (1 - self.b1) * gW
                self.vW[i] = self.b2 * self.vW[i] + (1 - self.b2) * (gW ** 2)
                self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * gb
                self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * (gb ** 2)
                mWh = self.mW[i] / (1 - self.b1 ** self.t)
                vWh = self.vW[i] / (1 - self.b2 ** self.t)
                mbh = self.mb[i] / (1 - self.b1 ** self.t)
                vbh = self.vb[i] / (1 - self.b2 ** self.t)
                l.W -= lr * mWh / (np.sqrt(vWh) + self.eps)
                l.b -= lr * mbh / (np.sqrt(vbh) + self.eps)
                i += 1


class Nadam(Adam):
    def step(self, layers, lr, wd=0.0):
        if self.mW is None:
            self._init(layers)
        self.t += 1
        i = 0
        for l in layers:
            if hasattr(l, "W"):
                gW = l.grad_W + wd * l.W
                gb = l.grad_b
                self.mW[i] = self.b1 * self.mW[i] + (1 - self.b1) * gW
                self.vW[i] = self.b2 * self.vW[i] + (1 - self.b2) * (gW ** 2)
                self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * gb
                self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * (gb ** 2)
                mWh = (self.b1 * self.mW[i] + (1 - self.b1) * gW) / (1 - self.b1 ** self.t)
                vWh = self.vW[i] / (1 - self.b2 ** self.t)
                mbh = (self.b1 * self.mb[i] + (1 - self.b1) * gb) / (1 - self.b1 ** self.t)
                vbh = self.vb[i] / (1 - self.b2 ** self.t)
                l.W -= lr * mWh / (np.sqrt(vWh) + self.eps)
                l.b -= lr * mbh / (np.sqrt(vbh) + self.eps)
                i += 1
