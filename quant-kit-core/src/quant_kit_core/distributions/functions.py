from typing import List, Tuple, Union
from warnings import warn

import numpy as np
import scipy.stats as stats
from numpy import ndarray

from quant_kit_core.distributions.containers import (
    CONTINUOUS_DISTRIBUTIONS,
    ContinuousDistribution,
)
from quant_kit_core.exceptions import TimeoutException
from quant_kit_core.utils import time_limit

__all__ = [
    "get_best_fit_distribution",
    "sample_gbm_returns",
]


def get_best_fit_distribution(
    data: ndarray,
    candidate_distributions: List[str] = None,
    support: Tuple[float, float] = None,
    n_restarts: int = 30,
    sample_size: int = 1000,
    return_scores: bool = False,
    fit_time_limit: int = 5,
) -> List[ContinuousDistribution]:
    """Given a 1D sample, find the best fit continuous distribution to model the sample

    Parameters
    ----------
    data: ndarray, shape=(N,)
        Data to estimate distribution for
    candidate_distributions: List[str], optional
        Names of distributions to use as candidates.
        If ``None`` then all continuous distributions with valid support will be used.
        (default = None)
    support: Tuple[float, float], optional
        Require support for a distribution to be considered valid.
        If ``None``, then the min, max of ``data`` will be used.
        (default = None)
    n_restarts: int, optional
        Number of restarts
        (default = 30)
    sample_size: int
        Sample size for restarts
        (default = 1000)
    return_scores: bool, optional
        Return matrix of earth mover's distance scores across restarts?
        (default = False)
    fit_time_limit: int, optional
        Execution time limit (seconds) for each distribution.
        (default = 5)
    """
    # Use the range of observations as the support
    if support is None:
        support = data.min(), data.max()

    if candidate_distributions is None:
        dists = [
            dist
            for dist in CONTINUOUS_DISTRIBUTIONS
            if dist.is_valid_for_range(*support)
        ]
    else:
        dists = [
            dist
            for name in candidate_distributions
            if (dist := ContinuousDistribution.from_name(name)).is_valid_for_range(
                *support
            )
        ]

    # Tabulate earth mover's distance for each candidate
    restarts = [np.random.choice(data, sample_size) for _ in range(n_restarts)]
    earth_mv_dists = []
    for dist in dists:
        try:
            with time_limit(fit_time_limit):
                earth_mv_dists.append(
                    [
                        stats.wasserstein_distance(
                            x_sample, dist.fit(x_sample).sample(size=sample_size)
                        )
                        for x_sample in restarts
                    ]
                )
        except TimeoutException:
            print(f"Fitting {dist.name} timed out")
            earth_mv_dists.append([np.nan] * n_restarts)
    earth_mv_dists = np.array(earth_mv_dists)

    # Filter out dists that failed to fit in time limit
    mask = np.isfinite(earth_mv_dists).all(-1)
    dists = np.array(dists)[mask]
    earth_mv_dists = earth_mv_dists[mask]

    # Avg the rank of each distribution's EMD in each restart
    avg_ranks = earth_mv_dists.argsort(0).argsort(0).mean(-1)

    # Order dists best to worst and fit to data
    ordering = avg_ranks.argsort()
    dists = [dist.fit(data) for dist in dists[ordering]]
    if return_scores:
        return dists, earth_mv_dists[ordering]
    else:
        return dists


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
