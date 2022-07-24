import numpy as np
from numpy import ndarray


__all__ = [
    "get_robust_trend_coef",
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
