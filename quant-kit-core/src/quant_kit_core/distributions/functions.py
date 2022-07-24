from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from numpy import ndarray


__all__ = [
    "sample_gbm_returns",
]


def sample_gbm_returns(
    drift: float,
    volatility: float,
    t: float,
    size: Union[int, Tuple[int, ...]],
    seed: Union[int, None] = None,
) -> ndarray:
    """Draw random samples from a geometric Brownian motion, arbitrage free 
    return distribution.

    Parameters
    ----------
    drift: float
        Annualized drift.
    volatility: float
        Annualized return standard deviation.
    t: float
        Length of return period, expressed as a fraction of a full year.
    size: Union[int, Tuple[int, ...]]: int
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.
    seed Union[int, None], optional
        Random state seed.
        (default = None)

    Returns
    -------
    returns: ndarray, shape=(n_samples,)
    """

    # Brownian motion
    rnd = np.random.RandomState(seed)
    wt = rnd.normal(0, np.sqrt(t), size)

    # geometric returns
    returns = np.exp((drift - volatility**2 / 2) * t + volatility * wt)
    return returns - 1
