import math


def mean_squared_error(y_true: float, y_pred: float) -> float:
    return 0.5 * (y_true - y_pred) ** 2

def binary_cross_entropy(y_true: float, y_pred: float) -> float:
    """
    y_true: القيمة الحقيقية (0 أو 1)
    y_pred: الناتج من الشبكة بعد sigmoid (بين 0 و 1)
    """
    # نضيف epsilon صغير عشان نتجنب log(0)
    eps = 1e-10
    y_pred = min(max(y_pred, eps), 1 - eps)

    loss = -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))
    return loss
