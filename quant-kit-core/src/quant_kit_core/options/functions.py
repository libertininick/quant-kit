from datetime import datetime
from typing import Dict, List, Union, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from scipy.stats import norm

from quant_kit_core.distributions.functions import sample_gbm_returns
from quant_kit_core.options.containers import (
    ContractType,
    Dividend,
    Option,
    OptionReplicationResult,
)
from quant_kit_core.utils import get_timediff, sample_gbm_returns


__all__ = [
    "get_price_bsm",
    "get_bsm_iv",
    "get_price_pcp",
    "get_price_statrep",
]


def get_implied_vol(
    price: float,
    option: Option,
    date: Union[datetime, str],
    spot: float,
    rf_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Solve for the Black-Scholes-Merton volatility implied by an option's price.

    Parameters
    ----------
    price: float
        Market price for option.
    option: Option
        Option contract to price.
    date: Union[datetime, str]
        Pricing date and time.
        Either a ``datetime`` or an ISO8601 formatted datetime string.
    spot: float
        Price at which the underlying security is trading at the moment of
        the Pricing date and time.
    rf_rate: float
        Annualized, continuously compounded risk free rate associated with
        time period between pricing date and the option's expiration date.
    dividend_yield: float, optional
        Annualized, continuously compounded dividend of the underlying.
        If the underlying doesn't pay any dividend, enter zero.
        (default = 0.)

    Returns
    -------
    volatility: float
        Annualized volatility of log return implied by an option's price assuming BSM.

    """

    def get_price_diff(vol: float) -> float:
        bsm_price = get_price_bsm(option, date, spot, vol, rf_rate, dividend_yield)
        return ((price - bsm_price) / spot) ** 0.5

    # Minimize replication errors
    result = minimize(
        fun=get_price_diff,
        x0=[0.1],
        bounds=[1e-4, 1e4],
        tol=1e-6,
    )


def get_price_bsm(
    option: Option,
    date: Union[datetime, str],
    spot: float,
    volatility: float,
    rf_rate: float,
    dividend_yield: float = None,
    dividends: Union[Dividend, List[Dividend], None] = None,
    dvd_rf_rates: Union[float, List[float], None] = None,
    n_samples: int = 10000,
) -> float:
    """Black-Scholes-Merton price for an option

    Parameters
    ----------
    option: Option
        Option contract to price.
    date: Union[datetime, str]
        Pricing date and time.
        Either a ``datetime`` or an ISO8601 formatted datetime string.
    spot: float
        Price at which the underlying security is trading at the moment of
        the Pricing date and time.
    volatility:
        Annualized return standard deviation expected to be realized for the
        underlying security over the remaining life of the option.
    rf_rate: float
        Annualized, continuously compounded risk free rate associated with
        time period between pricing date and the option's expiration date.
    dividend_yield: float, optional
        Annualized, continuously compounded dividend of the underlying.
        If the underlying doesn't pay any dividend, enter zero.
        If discrete dividend amounts and ex-dates are known, use ``dividends``
        and leave ``dividend_yield = None``.
        (default = None)
    dividends: Union[Dividend, List[Dividend], None]
        Discrete dividends to be that will be paid out during the term of the
        option contract.
        (default = None)
    dvd_rf_rates: Union[float, List[float], None]
        Risk free rates associated with each dividend.
    n_samples: int, optional
        Number of samples to use when simulating return paths for discrete
        dividend payments.
        (default = 10000)

    Returns
    -------
    price: float
        Option's price at Pricing date and time.


    Examples
    --------
    ### Single discrete dividend ###
    # See "Back to Basics: a new approach to the discrete dividend problem"
    # pg 22
    >>> option = Option(
    ...     type=ContractType.CALL, strike=130, expiration_date="2022-12-31"
    ... )

    >>> get_price_bsm(
    ...     option,
    ...     date="2021-12-31",
    ...     spot=100,
    ...     volatility=0.30,
    ...     rf_rate=0.06,
    ...     dividends=Dividend(amount=50, ex_date="2022-06-30"),
    ...     n_samples=1000000
    ... )
    0.23108108854408882

    ### Multiple discrete dividends ###
    # See "Back to Basics: a new approach to the discrete dividend problem"
    # pgs 25 - 27
    >>> dvds = [
    ...     Dividend(amount=4, ex_date="2022-06-30"),
    ...     Dividend(amount=4, ex_date="2023-06-30"),
    ...     Dividend(amount=4, ex_date="2024-06-30")
    ... ]

    >>> option = Option(
    ...     type=ContractType.CALL, strike=100, expiration_date="2024-12-31"
    ... )

    >>> for dt in ["2023-12-31", "2022-12-31", "2021-12-31"]:
    ...     print(
    ...         get_price_bsm(
    ...             option,
    ...             date=dt,
    ...             spot=100,
    ...             volatility=0.25,
    ...             rf_rate=0.06,
    ...             dividends=[dvd for dvd in dvds if dvd.ex_date > dt],
    ...             n_samples=1000000
    ...         )
    ...     )
    10.681185245211447
    15.174078038306856
    18.585904904092878
    """
    # Time to expiration
    t = get_timediff(date, option.expiration_date)

    if t <= 0:
        raise ValueError("Pricing date and time is on or after expiration")

    if dividends is not None:
        if isinstance(dividends, Dividend):
            dividends = [dividends]

        if dvd_rf_rates is not None:
            if isinstance(dvd_rf_rates, float):
                dvd_rf_rates = [dvd_rf_rates] * len(dividends)

            assert len(dvd_rf_rates) == len(dividends)
        else:
            dvd_rf_rates = [rf_rate] * len(dividends)

        # Get time slice schedule based on dividend ex-dates:
        # [ex-1, ex-2, ..., option expiration]
        t_schedule = np.array(
            [get_timediff(date, dvd.ex_date) for dvd in dividends] + [t]
        )
        t_diffs = np.diff(t_schedule, prepend=0)
        assert np.allclose(t_diffs.sum(), t)

        # Get marginal discount rates based on time slice schedule:
        # [rate 1 to 2, rate 2 to 3, ..., rate N-1 to N]
        rate_schedule = np.array(dvd_rf_rates + [rf_rate])
        rate_schedule = np.exp(-rate_schedule * t_schedule)
        marginal_rates = np.concatenate(
            (rate_schedule[:1], rate_schedule[1:] / rate_schedule[:-1]), axis=-1
        )
        assert np.allclose(marginal_rates.cumprod(), rate_schedule)
        marginal_rates = np.log(marginal_rates) / -t_diffs

        # Sequence of dividend payments
        dvd_pmts = [dvd.amount for dvd in dividends] + [0.0]

        # Simulate return paths in (# dvd payments + 1) slices
        for pmt_i, rfr_i, t_i in zip(dvd_pmts, marginal_rates, t_diffs):
            rets_i = sample_gbm_returns(
                drift=rfr_i,
                volatility=volatility,
                t=t_i,
                size=n_samples,
                seed=int((t_i % 1) * 2**32),
            )

            # Grow spot by returns and then remove discrete dividend payment
            # from the new price. If dividend payment is larger than equity,
            # pay out as much as possible in a liquidation.
            spot = np.maximum(spot * (1 + rets_i) - pmt_i, 0)

        # Find future expected value based on contract type
        if option.type is ContractType.CALL:
            ev = np.maximum(spot - option.strike, 0).mean()
        elif option.type is ContractType.PUT:
            ev = np.maximum(option.strike - spot, 0).mean()
        else:
            raise NotImplementedError(f"ContractType {option.type} not implemented")

        # Present value of future expected value
        price = ev * np.exp(-rf_rate * t)
    else:
        if dividend_yield is None:
            dividend_yield = 0

        # Black-Scholes d1 and d2
        d1 = (
            np.log(spot / option.strike)
            + t * (rf_rate - dividend_yield + volatility**2 / 2)
        ) / (volatility * t**0.5)

        d2 = d1 - volatility * t**0.5

        # Present value of spot and strike
        spot_pv = spot * np.exp(-dividend_yield * t)
        strike_pv = option.strike * np.exp(-rf_rate * t)

        # Find price based on contract type
        if option.type is ContractType.CALL:
            price = spot_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2)
        elif option.type is ContractType.PUT:
            price = strike_pv * norm.cdf(-d2) - spot_pv * norm.cdf(-d1)
        else:
            raise NotImplementedError(f"ContractType {option.type} not implemented")

    return price


def get_price_pcp(
    price: float,
    option: Option,
    date: Union[datetime, str],
    spot: float,
    rf_rate: float,
    dividend_yield: float = None,
    dividends: Union[Dividend, List[Dividend], None] = None,
    dvd_rf_rates: Union[float, List[float], None] = None,
) -> float:
    """Price via put-call parity

    No dividends: `C - P = S - K*exp(-r*t))`
    Discrete dividends: `C - P = S - PV(Div) - K*exp(-r*t)`
    Continuous dividend: `C - P = S*exp(-y*t) - K*exp(-r*t)`

    Parameters
    ----------
    price: float
        Price of the corresponding put or call with same strike and expiration date.
    option: Option
        Option contract to price.
    date: Union[datetime, str]
        Pricing date and time.
        Either a ``datetime`` or an ISO8601 formatted datetime string.
    spot: float
        Price at which the underlying security is trading at the moment of
        the Pricing date and time.
    rf_rate: float
        Annualized, continuously compounded risk free rate associated with
        time period between pricing date and the option's expiration date.
    dividend_yield: float, optional
        Annualized, continuously compounded dividend of the underlying.
        If the underlying doesn't pay any dividend, enter zero.
        If discrete dividend amounts and ex-dates are known, use ``dividends``
        and leave ``dividend_yield = None``.
        (default = None)
    dividends: Union[Dividend, List[Dividend], None]
        Discrete dividends to be that will be paid out during the term of the
        option contract.
        (default = None)
    dvd_rf_rates: Union[float, List[float], None]
        Risk free rates associated with each dividend.

    Returns
    -------
    pcp_price: float
        Option's price at pricing date and time.
    """
    # Time to expiration
    t = get_timediff(date, option.expiration_date)

    if t <= 0:
        raise ValueError("Pricing date and time is on or after expiration")

    # Present value of strike price
    strike_pv = option.strike * np.exp(-rf_rate * t)

    # Adjust spot for any dividends
    if dividends is not None:

        if isinstance(dividends, Dividend):
            dividends = [dividends]

        if dvd_rf_rates is not None:
            if isinstance(dvd_rf_rates, float):
                dvd_rf_rates = [dvd_rf_rates] * len(dividends)

            assert len(dvd_rf_rates) == len(dividends)
        else:
            dvd_rf_rates = [rf_rate] * len(dividends)

        # Adjust spot by present value of all discrete dividends
        spot -= sum(
            [
                dvd.amount * np.exp(-rfr * get_timediff(date, dvd.ex_date))
                for dvd, rfr in zip(dividends, dvd_rf_rates)
            ]
        )
    elif dividend_yield is not None:
        # Adjust spot by discount factor of continuous dividend yield
        spot *= np.exp(-dividend_yield * t)

    # Adjusted spot - present value of strike
    spot_strike_diff = spot - strike_pv

    if option.type is ContractType.CALL:
        pcp_price = spot_strike_diff + price
    elif option.type is ContractType.PUT:
        pcp_price = price - spot_strike_diff
    else:
        raise ValueError("Put-Call Parity is not defined for other contract types")

    return pcp_price


def get_price_statrep(
    returns: ndarray,
    timediff: float,
    strike: float,
    spot: float,
    rf_rate: float,
    dividend_yield: float = 0.0,
) -> OptionReplicationResult:
    """Price a European call and put option of the same strike using static,
    statistical replication from a forecasted return distribution at option expiration.

    Given the expected return distribution at expiration, goal is to engineer a
    static hedge using stock and borrowing/lending at the risk-free-rate to minimize
    the square errors of selling an option and implementing the hedge.

    Parameters
    ----------
    returns: ndarray, shape=(N,)
        Forecasted (total) return distribution at option expiration.
    timediff: float, range=(0.0, inf)
        Total time difference between pricing date and expiration date, expressed
        as a fraction of a calendar year.
    strike: float
        Strike price to price call and put at.
    spot: float
        Price at which the underlying security is trading at the moment of
        the pricing date and time.
    rf_rate: float
        Annualized, continuously compounded risk free rate associated with
        time period between pricing date and the option's expiration date.
    dividend_yield: float, optional
        Annualized, continuously compounded dividend of the underlying.
        If the underlying doesn't pay any dividend, enter zero.
        (default = 0)

    Returns
    -------
    OptionReplicationResult
    """

    # Future price distribution; Adjust spot price for continuous dividend yield
    future_prices = spot * np.exp(-dividend_yield * timediff) * (1 + returns)

    # Option payoffs
    call_payoffs = np.maximum(0, future_prices - strike)
    put_payoffs = np.maximum(0, strike - future_prices)

    # Financing cost per unit long of underlying (negative cost implies income)
    financing_cost = spot * (np.exp(timediff * (rf_rate - dividend_yield)) - 1)

    # Long underlying payoffs
    long_payoffs = future_prices - spot - financing_cost

    # Functions for replication error minimization
    def get_hedging_pnls(h: float) -> Tuple[ndarray, ndarray]:
        # Short call + long hedge PnL
        hedged_call_pnls = h * long_payoffs - call_payoffs

        # Short put + short hedge PnL
        hedged_put_pnls = (h - 1) * long_payoffs - put_payoffs

        return hedged_call_pnls, hedged_put_pnls

    def get_hedging_error(params) -> float:
        hedged_call_pnls, hedged_put_pnls = get_hedging_pnls(*params)

        # Mean squared error
        mse = ((hedged_call_pnls / spot) ** 2 + (hedged_put_pnls / spot) ** 2).mean()

        return mse

    # Minimize replication errors
    result = minimize(
        fun=get_hedging_error,
        x0=[0.5],
        bounds=[(0.0, 1.0)],
        tol=1e-6,
    )

    if result.success:
        # Optimal parameters
        h = result.x[0]
        error = result.fun

        # PnLs of hedged portfolios
        hedged_call_pnls, hedged_put_pnls = get_hedging_pnls(h)

        # Future expected value of call and put
        call_ev = -hedged_call_pnls.mean()
        put_ev = -hedged_put_pnls.mean()

        # Discount back to get option prices
        discount_factor = np.exp(timediff * -rf_rate)
        call = call_ev * discount_factor
        put = put_ev * discount_factor

        return ReplicationResult(call, put, h, error)
    else:
        raise RuntimeError("Minimization of replication errors was unsuccessful.")
