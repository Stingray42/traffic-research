from datetime import datetime
from typing import Callable

import numpy as np


def generator(distribution: str, seed: int = None, **kwargs):
    if 'size' in kwargs:
        kwargs.pop('size')
    if seed is None:
        seed = int(datetime.now().timestamp())
    rng = np.random.default_rng(seed)
    return lambda n: getattr(rng, distribution)(size=n, **kwargs)


def secant(f: Callable[[float], float], x0: float, eps: float = 1e-7, kmax: int = 1e3) -> float:
    """
    solves f(x) = 0 by secant method with precision eps
    :param f: f
    :param x0: starting point
    :param eps: precision wanted
    :param kmax: max iterations
    :return: root of f(x) = 0
    """
    x, x_prev, i = x0, x0 + 2 * eps, 0

    while abs(x - x_prev) >= eps and i < kmax:
        x, x_prev, i = x - f(x) / (f(x) - f(x_prev)) * (x - x_prev), x, i + 1

    return x
