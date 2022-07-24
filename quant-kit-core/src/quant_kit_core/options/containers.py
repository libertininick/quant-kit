from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Union


__all__ = [
    "ContractType",
    "Dividend",
    "Option",
    "OptionReplicationResult",
]


class ContractType(Enum):
    CALL = auto()
    PUT = auto()


@dataclass(frozen=True)
class Dividend:
    """Future discrete dividend payment

    Attributes
    ----------
    amount: float
        Dividend amount
    ex_date: Union[datetime, str]
        Date security goes ex-dividend
        Either a ``datetime`` or an ISO8601 formatted datetime string.
    """

    amount: float
    ex_date: Union[datetime, str]


@dataclass(frozen=True)
class Option:
    """
    Attributes
    ----------
    type: ContractType
        Contract type (put, call, ...).
    strike: float
        Option's strike price.
    expiration_date:
        Option's expiration date.
        Either a ``datetime`` or an ISO8601 formatted datetime string.

    """

    type: ContractType
    strike: float
    expiration_date: Union[datetime, str]


@dataclass(frozen=True)
class OptionReplicationResult:
    """
    Attributes
    ----------
    call: float
        Call price.
    put: float
        Put price.
    h: float, range=[0., 1.]
        Hedge ratio.
    error: float
        Replication error at optimal hedge ratio.
    """

    call: float
    put: float
    h: float
    error: float
