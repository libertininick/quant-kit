"""
Package wide utility functions
"""
import signal
from calendar import isleap
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Union

from quant_kit_core.exceptions import TimeoutException

__all__ = [
    "get_timediff",
    "time_limit",
]


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


@contextmanager
def time_limit(seconds: int):
    """Set an execution time limit for a function call

    Parameters
    ----------
    seconds: int

    Examples
    --------
    >>> try:
    ...     with time_limit(10):
    ...         long_function_call()
    ... except TimeoutException as e:
    ...     print("Timed out!")
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
