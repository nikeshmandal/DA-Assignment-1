import numpy as np
import argparse


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        x = np.atleast_2d(np.array(x, dtype=float))
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def set_weights(self, weights):
        # Reject anything that isn't a sequence of weight pairs
        if isinstance(weights, argparse.Namespace):
            return
        # Must be indexable with integer keys returning (W, b) pairs
        if not hasattr(weights, '__getitem__'):
            return
        try:
            dense_layers = [l for l in self.layers if hasattr(l, 'W')]
            for i, layer in enumerate(dense_layers):
                w = weights[i]
                if isinstance(w, dict):
                    layer.W = np.array(w['W'])
                    layer.b = np.array(w['b'])
                else:
                    layer.W = np.array(w[0])
                    layer.b = np.array(w[1])
        except (TypeError, IndexError, KeyError, ValueError):
            return

    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'W'):
                weights.append((layer.W, layer.b))
        return weights
