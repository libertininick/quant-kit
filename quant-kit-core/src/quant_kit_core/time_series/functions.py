import numpy as np
from numpy import ndarray


__all__ = [
    "get_robust_trend_coef",
    "get_rolling_windows",
]


def get_robust_trend_coef(x: ndarray, n_bins: int = 10) -> float:
    """Calculates the trend coefficient for a sequence of observations

    Parameters
    ----------
    x: ndarray, shape=(N,)
        Time series to calculate trend coefficient for.
    n_bins: int, optional
        Number of equal sized segments to break series into and reduce via median.
        (default = 10)

    Returns
    -------
    trend_coef: float, range=[-1., 1.]
        Time series' trend coefficient.
    """
    n = len(x)
    n_trunc = (n // n_bins) * n_bins

    # Split series into equal sized bins
    idxs = np.sort(np.random.choice(np.arange(n), n_trunc, replace=False))
    splits = np.array(np.split(idxs, n_bins))

    # Calculate median value of bins (removes outliers)
    split_medians = np.median(x[splits], axis=-1)

    # Rank values smallest to largest
    ranks = np.argsort(np.argsort(split_medians))

    # Perfect up-trend (monotonically increasing ranks)
    perfect_up_trend = np.arange(n_bins)

    trend_coef = np.corrcoef(perfect_up_trend, ranks)[0, 1]

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
