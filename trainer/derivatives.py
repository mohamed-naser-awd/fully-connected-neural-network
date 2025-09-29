from network.activation import sigmoid


def d_relu(z: float) -> float:
    return 1.0 if z > 0 else 0.0


def d_mse(y_true: float, y_pred: float):
    return y_pred - y_true


def d_sigmoid(x: float) -> float:
    s = sigmoid(x)
    return s * (1 - s)


def d_binary_cross_entropy(y_true: float, y_pred: float) -> float:
    """
    Gradient of Binary Cross Entropy loss w.r.t y_pred
    y_true: القيمة الحقيقية (0 أو 1)
    y_pred: الناتج من الشبكة بعد sigmoid (بين 0 و 1)
    """
    eps = 1e-10
    y_pred = min(max(y_pred, eps), 1 - eps)

    return (y_pred - y_true) / (y_pred * (1 - y_pred))


d_loss = d_binary_cross_entropy
