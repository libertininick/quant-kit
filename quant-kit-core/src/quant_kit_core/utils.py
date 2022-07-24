"""
Package wide utility functions
"""
from calendar import isleap
from datetime import datetime, timedelta
from typing import List, Tuple, Union


__all__ = [
    "get_timediff",
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
