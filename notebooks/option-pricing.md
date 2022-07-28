---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python [conda env:quant_kit_env]
    language: python
    name: conda-env-quant_kit_env-py
---

```python
%load_ext autoreload
%autoreload 2

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_kit_core.distributions import sample_gbm_returns
from quant_kit_core.options import (
    ContractType, 
    Dividend, 
    Option,
    get_price_bsm,
    get_price_pcp,
    get_price_statrep
)
from quant_kit_core.utils import get_timediff


# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

# Test pricing methods

```python
strike = 100
expiration_date = "2022-12-31"

call_option = Option(ContractType.CALL, strike, expiration_date)

put_option = Option(ContractType.PUT, strike, expiration_date)
```

```python
date = "2022-04-30"
spot = 100
volatility = 0.276
rf_rate = 0.05
dvd_yld = 0.035
t = get_timediff(date, expiration_date)

future_returns = sample_gbm_returns(
    drift=rf_rate - dvd_yld, 
    volatility=volatility, 
    t=t, 
    size=1000000
)
future_prices = spot * (1 + future_returns)
print(f"Mean   ${future_prices.mean():>10.4f}\nMedian ${np.median(future_prices):>10.4f}")

fig, ax = plt.subplots(figsize=(10,5))
ax.hist(future_prices, bins=50, edgecolor="black", alpha=0.5)
_ = ax.set_title("GBM Future price distribution")
```

```python
call_price_bsm = get_price_bsm(call_option, date, spot, volatility, rf_rate, dvd_yld)

put_price_bsm = get_price_bsm(put_option, date, spot, volatility, rf_rate, dvd_yld)

call_price_pcp = get_price_pcp(
    put_price_bsm, call_option, date, spot, rf_rate, dvd_yld
)

print(f"Call BSM: ${call_price_bsm:>10.6f}\nPut BSM : ${put_price_bsm:>10.6f}\nCall PCP: ${call_price_pcp:>10.6f}")
```

```python
get_statrep_prices(
    returns=sample_gbm_returns(rf_rate, volatility, t, size=1000000),
    timediff=t,
    strike=strike,
    spot=spot,
    rf_rate=rf_rate,
    dividend_yield=dvd_yld,
)
```

```python
option = Option(
    type=ContractType.CALL,
    strike=100,
    expiration_date="2024-12-31"
)

dvds = [
    Dividend(amount=4, ex_date="2022-06-30"),
    Dividend(amount=4, ex_date="2023-06-30"),
    Dividend(amount=4, ex_date="2024-06-30")
]


for dt in ["2023-12-31", "2022-12-31", "2021-12-31"]:
    print(
        get_bsm_price(
            option,
            date=dt,
            spot=100,
            volatility=0.25,
            rf_rate=0.06,
            dividends=[dvd for dvd in dvds if dvd.ex_date > dt],
            n_samples=1000000
        )
    )
```

<!-- #region heading_collapsed=true -->
# BSM vs statrep
<!-- #endregion -->

```python hidden=true
date = "2022-04-30"
spot = 100
expiration_dates = pd.date_range(start=date, periods=30, freq='D').tolist()[1:]
# expiration_dates += pd.date_range(start=max(expiration_dates), periods=30, freq='W').tolist()[1:]
# expiration_dates += pd.date_range(start=max(expiration_dates), periods=13, freq='m').tolist()[1:]
expiration_dates = [
    pd.Timestamp.to_pydatetime(dt) for dt in expiration_dates
]
strikes = np.linspace(70, 130, 11)
vols = np.linspace(0.03, 1.5, 10)
rf_rates = np.linspace(0.0, 0.25, 5)
dvd_ylds = np.linspace(0.0, 0.25, 5)
```

```python hidden=true
outputs = []
for ex_dt in expiration_dates:
    for k in strikes:
        abs_moneyness = np.abs(k - spot)
        call_option = Option(ContractType.CALL, k, ex_dt)
        put_option = Option(ContractType.PUT, k, ex_dt)
        t = get_timediff(date, ex_dt)
        for vol in vols:
            price_std = spot * (np.exp(vol ** 0.5) - 1)
            std_moneyness = abs_moneyness / price_std
            for rfr in rf_rates:
                for dvd_yld in dvd_ylds:
                    call_price_bsm = get_bsm_price(call_option, date, spot, vol, rfr, dvd_yld)
                    put_price_bsm = get_bsm_price(put_option, date, spot, vol, rfr, dvd_yld)
                    result = get_statrep_prices(
                        returns=sample_gbm_returns(rfr, vol, t, size=100000),
                        timediff=t,
                        strike=k,
                        spot=spot,
                        rf_rate=rfr,
                        dividend_yield=dvd_yld,
                    )
                    outputs.append(
                        dict(
                            timediff=t, 
                            abs_moneyness=abs_moneyness,
                            std_moneyness=std_moneyness,
                            vol=vol,
                            rfr=rfr,
                            dvd_yld=dvd_yld,
                            call_bsm=call_price_bsm,
                            call_sim=result.call, 
                            call_err=(call_price_bsm - result.call) / price_std,
                            put_bsm=put_price_bsm, 
                            put_sim=result.put,
                            put_err=(put_price_bsm - result.put) / price_std,
                            hedge_ratio=result.h,
                            result_err=result.error,
                        )
                    )
outputs = pd.DataFrame(outputs)
```

```python hidden=true
pd.qcut()
```

```python hidden=true
outputs["hedge_bin"] = pd.qcut(outputs.hedge_ratio, q=np.linspace(0,1,11), labels=False)
outputs["result_err_bin"] = pd.qcut(outputs.result_err, q=np.linspace(0,1,11), labels=False)
```

```python hidden=true
for grp, grp_df, in outputs.groupby("result_err_bin"):
    print(f"{grp:>10}, {np.abs(grp_df.call_err).mean():>10.6f} {np.abs(grp_df.put_err).mean():>10.6f}")
```

```python hidden=true
fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(outputs.vol, outputs.hedge_ratio)

```

```python hidden=true
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(outputs.hedge_ratio, bins=100)
```

```python hidden=true
z = np.log(prices[:,3]/prices[:,1])
```

```python hidden=true
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(z)
```

```python hidden=true
fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(prices[:,1], prices[:,3])
```

# Non-normal distributions

```python
strike = 100
expiration_date = "2022-06-30"

call_option = Option(ContractType.CALL, strike, expiration_date)
put_option = Option(ContractType.PUT, strike, expiration_date)

date = "2022-04-30"
spot = 100
rf_rate = 0.005
dvd_yld = 0.0
t = get_timediff(date, expiration_date)

log_returns = np.concatenate(
    (
        np.random.randn(10000) * 0.01 - 0.19,
        np.random.randn(1000) * 0.01,
        np.random.randn(9000) * 0.01 + 0.19,
    ),
)

future_prices = spot * np.exp(-dvd_yld * t) * np.exp(log_returns)
drift = np.log(future_prices / spot).mean()
vol_eqv = ((log_returns**2).mean() / t)**0.5

gbm_returns = sample_gbm_returns(
    drift=rf_rate - dvd_yld, 
    volatility=vol_eqv, 
    t=t, 
    size=len(log_returns)
)
gbm_prices = spot * np.exp(gbm_returns)
gbm_drift = np.log(gbm_prices / spot).mean()

print(f"Drift: {drift:.2%}\nGBM Drift: {gbm_drift:.2%}\nVol Eqv: {vol_eqv:.2%}\n")

fig, ax = plt.subplots(figsize=(10,5))
_, bins, _ = ax.hist(
    gbm_prices, 
    bins=np.linspace(
        min(future_prices.min(), gbm_prices.min()), 
        max(future_prices.max(), gbm_prices.max()),
        52
    ),
    edgecolor="black",
    color="black",
    alpha=0.25
)
_ = ax.hist(
    future_prices, 
    bins=bins, 
    edgecolor="black",
    alpha=0.5,
)

call_price_bsm = get_bsm_price(call_option, date, spot, vol_eqv, rf_rate, dvd_yld)
put_price_bsm = get_bsm_price(put_option, date, spot, vol_eqv, rf_rate, dvd_yld)
print(f"Call BSM: ${call_price_bsm:>10.6f}\nPut BSM : ${put_price_bsm:>10.6f}")

result = get_statrep_prices(
    returns=np.exp(log_returns) - 1,
    timediff=t,
    strike=strike,
    spot=spot,
    rf_rate=rf_rate,
    dividend_yield=dvd_yld,
)
print(f"h: {result.h:.2%}\nCall rep: ${result.call:>10.6f}\nPut rep : ${result.put:>10.6f} ")
```

```python
spot
```

```python
spot*(np.exp((rf_rate - dvd_yld)*t) - 1)
```

```python
(future_prices - spot - spot*(np.exp((rf_rate - dvd_yld)*t) - 1)).mean()
```

```python
np.maximum(0, strike - future_prices).mean()
```

```python
pnls = (future_prices - spot - spot*(np.exp((rf_rate - dvd_yld)*t) - 1))*result.h1 - np.maximum(0, future_prices - strike)

pnls = (future_prices - spot - spot*(np.exp((rf_rate - dvd_yld)*t) - 1))*result.h2 - np.maximum(0, strike - future_prices)

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(
    pnls, #+ 13.874897*np.exp(t*rf_rate), 
    bins=50, 
    edgecolor="black", 
    alpha=0.5
)

```

```python
(pnls + 39.227354*np.exp(t*rf_rate)).mean()
```

```python
replication_prices
```

```python
get_pcp_price(
    put_option,
    date,
    replication_prices.call,
    spot,
    rf_rate,
    dvd_yld,
)


```

```python
spot*np.exp(-dvd_yld*t) - strike*np.exp(-rf_rate*t)
```

```python

result
```

```python
replication_payoffs = get_replication_payoffs(future_prices, strike, *result.x)

discount_factor = np.exp(timediff * -rf_rate)
*discount_factor, replication_payoffs.put.mean()*discount_factor
```

```python
type(result)
```

```python
call_replicating_payoffs.mean() * , put_replicating_payoffs.mean() * np.exp(timediff * -rf_rate)
```

```python
spot * np.exp(-timediff * dvd_yld) * (call_hedge_ratio - borrow_factor)
```

```python
short_call_pnl, short_put_pnl = get_replicating_pnls(result.x[0])
```

```python
call_ev, put_ev = get_expected_values(short_call_pnl, short_put_pnl)
print(f"${call_ev:>10.4f}, ${put_ev:>10.4f}")
print(f"${max(0, spot - strike):>10.4f}, ${max(0, strike - spot):>10.4f}")
```

```python
xs = np.linspace(0,1,100)
ys = [get_replicating_errors(hr) for hr in xs]
fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(xs,ys)
```

```python
np.ap
```
