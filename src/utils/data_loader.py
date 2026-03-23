import numpy as np
from sklearn.datasets import fetch_openml

def load_dataset(name):
    if name == "mnist":
        data = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="liac-arff"
        )
    elif name == "fashion_mnist":
        data = fetch_openml(
            "Fashion-MNIST",
            version=1,
            as_frame=False,
            parser="liac-arff"
        )
    else:
        raise ValueError("dataset must be mnist or fashion_mnist")

    X = data.data.astype(np.float32) / 255.0
    y = data.target.astype(int)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    return X_train, y_train, X_test, y_test