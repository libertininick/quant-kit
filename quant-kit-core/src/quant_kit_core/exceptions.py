"""
Module exceptions
"""


class BaseException(Exception):
    pass


class TimeoutException(BaseException):
    """
    Execution time limit reached
    """
