from network.activation import sigmoid


def d_mse(y_true: float, y_pred: float):
    return y_pred - y_true


def d_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)
