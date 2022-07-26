"""
Package wide utility functions
"""
from calendar import isleap
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray


__all__ = [
    "get_rolling_windows",
    "get_timediff",
]


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


def get_timediff(dt1: Union[datetime, str], dt2: Union[datetime, str]) -> float:
    """Calculates the total time difference between two dates, expressed as a
    fraction of a calendar year.

    NOTE: dt2's year is used to determine the total number of days in a year.

    Parameters
    ----------
    dt1: Union[datetime, str]
        First date.
        Either a ``datetime`` or an ISO8601 formatted datetime string.

    dt2: Union[datetime, str]
        Second (later) date.
        Either a ``datetime`` or an ISO8601 formatted datetime string.

    Returns
    -------
    diff: float
        Time difference.
    """
    dt1 = datetime.fromisoformat(dt1) if isinstance(dt1, str) else dt1
    dt2 = datetime.fromisoformat(dt2) if isinstance(dt2, str) else dt2

    days_in_yr = 365 + int(isleap(dt2.year))
    return (dt2 - dt1) / timedelta(days=1) / days_in_yr
