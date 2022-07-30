from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import scipy.stats as stats
from numpy import ndarray
from numpy.random import RandomState

__all__ = ["CONTINUOUS_DISTRIBUTIONS", "ContinuousDistribution"]

# Name and reasonable parameters for all scipy probability distributions
_DIST_PARAMS: Dict[str, Tuple[float, ...]] = dict(stats._distr_params.distcont)

# All scipy continuous distributions
_CONTINUOUS_DISTRIBUTIONS = dict(
    sorted(
        [
            (name, (dist, dist.support(*_DIST_PARAMS[name])))
            for name, dist in stats.distributions.__dict__.items()
            if isinstance(dist, stats.rv_continuous) and name in _DIST_PARAMS
        ],
        key=lambda x: x[0],
    )
)


@dataclass
class ContinuousDistribution:
    """Container for continuous distribution"""

    name: str
    dist: stats.rv_continuous = field(repr=False)
    support: Tuple[float, float]
    params: List[float] = None

    @property
    def is_fit(self) -> bool:
        "Check whether the distribution has fitted parameters"
        return self.params is not None

    def __post_init__(self):
        if self.params is not None:
            self.dist = self.dist(*self.params)

    def cdf(self, x: ndarray, *args, **kwargs) -> ndarray:
        """Cumulative distribution function (CDF) at ``x``.

        Parameters
        ----------
        x : ndarray
            Quantiles

        Returns
        -------
        cdf: ndarray
            Cumulative distribution function evaluated at ``x``
        """
        return self.dist.cdf(x, *args, **kwargs)

    def fit(
        self, data: np.ndarray, **kwargs: Dict[str, Any]
    ) -> "ContinuousDistribution":
        """Estimate distribution's parameters from data

        Parameters
        ----------
        data: np.ndarray
            Data to fit distribution to.
        **kwargs
            Named arguments to pass to ``self.dist.fit``

        Returns
        -------
        ContinuousDistribution
            A fit distribution.
        """
        return ContinuousDistribution(
            name=self.name,
            dist=self.dist,
            support=self.support,
            params=self.dist.fit(data, **kwargs),
        )

    def is_valid_for_range(self, a_min: float, a_max: float) -> bool:
        "Checks distribution's support for a given range"
        s_lower, s_upper = self.support
        return a_min >= s_lower and a_max <= s_upper

    def pdf(self, x: ndarray, *args, **kwargs) -> ndarray:
        """Probability density function (PDF) at ``x``.

        Parameters
        ----------
        x : ndarray
            Quantiles

        Returns
        -------
        pdf: ndarray
            Probability density function evaluated at ``x``
        """
        return self.dist.pdf(x, *args, **kwargs)

    def ppf(self, q: ndarray, *args, **kwargs) -> ndarray:
        """Percent point function (inverse of ``cdf``) at ``q``.

        Parameters
        ----------
        q : ndarray
            Lower tail probability

        Returns
        -------
        pdf: ndarray
            Quantile corresponding to the lower tail probability `q`.
        """
        return self.dist.ppf(q, *args, **kwargs)

    def sample(
        self, size: Union[int, Tuple[int, ...]] = (1,), random_state: RandomState = None
    ) -> np.ndarray:
        """Take a random sample from the distribution

        Parameters
        ----------
        size: Union[int, Tuple[int,...]]
            Sample size.
        random_state: RandomState
            Random state

        Returns
        -------
        sample: ndarray, shape=size
        """
        return self.dist.rvs(size, random_state)

    @classmethod
    def from_name(cls, name: str) -> "ContinuousDistribution":
        if name in _CONTINUOUS_DISTRIBUTIONS:
            return ContinuousDistribution(name, *_CONTINUOUS_DISTRIBUTIONS[name])
        else:
            raise ValueError(f"{name} is not a known scipy continuous distribution")


CONTINUOUS_DISTRIBUTIONS = [
    ContinuousDistribution.from_name(name) for name in _CONTINUOUS_DISTRIBUTIONS
]
