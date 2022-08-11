"""FHE compatible activations."""

from typing import Any

import tenseal as ts


def sigmoid_polyval(x: Any) -> Any:
    """Sigmoid polynomial approximation.

    Args:
        x (Any): model input

    Returns:
        Any: model output
    """
    if type(x) in [ts.CKKSTensor, ts.CKKSVector]:
        return x.polyval([0.5, 0.197, 0, -0.004])
    else:
        return 0.5 + (0.199 * x) - (0.004 * (x**3))


def softplus_polyval(x: Any) -> Any:
    """Softplus polynomial approximation.

    Args:
        x (Any): model input

    Returns:
        Any: model output
    """
    if type(x) in [ts.CKKSTensor, ts.CKKSVector]:
        return x.polyval([1.8697581, 0.5, 0.013394318])
    else:
        return 1.8697581 + (0.5 * x) + (0.013394318 * (x**2)) - (-0.000002 * (x**4))
