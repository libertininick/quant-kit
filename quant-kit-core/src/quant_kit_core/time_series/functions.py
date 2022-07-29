import numpy as np
from numpy import ndarray


__all__ = [
    "get_robust_trend_coef",
    "get_rolling_windows",
]


def get_robust_trend_coef(returns: ndarray) -> float:
    """Calculates the trend coefficient for a sequence of observations

    Parameters
    ----------
    returns: ndarray, shape=(N,)
        Return series to calculate trend coefficient for.
        NOTE: Assumes log-returns.

    Returns
    -------
    trend_coef: float, range=[-1., 1.]
        Time series' trend coefficient.
    """
    if returns.ndim != 1:
        raise ValueError("Only 1D arrays are supported")

    # Get rolling windows of length=SQRT(N)
    windows = get_rolling_windows(returns, window_size=int(len(returns) ** 0.5))

    # Median return of each window
    medians = np.median(windows, axis=-1)

    # Cumulative returns
    cumrets = np.cumsum(medians)

    # Perfect up-trend (monotonically increasing cumulative return)
    perfect_up_trend = np.arange(len(cumrets))

    trend_coef = np.corrcoef(perfect_up_trend, cumrets)[0, 1]

    return trend_coef


def get_rolling_windows(arr: ndarray, window_size: int, stride: int = 1) -> ndarray:
    """Generate a rolling window of input based on a target window and step size

    Parameters
    ----------
    arr: ndarray, shape=(N,...)
        A n-dim array
    window_size: int
        Rolling window size.
    stride: int, optional
        Window step size.
        (default = 1)

    Returns
    -------
    windows: ndarray, shape=((N - window_size) // stride + 1, window_size, ...)
        Rolling windows from input.

    Examples
    --------
    >>> x = np.arange(10)
    >>> get_rolling_windows(x, window_size=3)
    array([
       [0, 1, 2],
       [1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6],
       [5, 6, 7],
       [6, 7, 8],
       [7, 8, 9]
    ])

    >>> get_rolling_windows(x, window_size=3, stride=2)
    array([
       [0, 1, 2],
       [2, 3, 4],
       [4, 5, 6],
       [6, 7, 8]
    ])

    >>> x = np.random.rand(37,5)
    >>> windows = get_rolling_windows(x, window_size=6, stride=3)
    >>> windows.shape
    (11, 6, 5)
    """
    shape = (arr.shape[0] - window_size + 1, window_size) + arr.shape[1:]
    strides = (arr.strides[0],) + arr.strides
    rolled = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], stride)]
