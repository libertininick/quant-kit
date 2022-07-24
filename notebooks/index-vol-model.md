---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

```python
%load_ext autoreload
%autoreload 2

import glob
import json
from collections import defaultdict
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg as pg
import scipy.stats as stats
import torch
import torch.nn as nn
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from torch import Tensor

from quant_kit_core import get_timediff
from quant_kit_core.distributions import sample_gbm_returns
from quant_kit_core.options import (
    ContractType, 
    Dividend, 
    Option,
    get_price_bsm,
    get_price_pcp,
    get_price_statrep
)
from quant_kit_core.time_series import get_robust_trend_coef

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

# Load Data

```python
engine = pg.connect(
    user='admin',
    password='quest',
    host='127.0.0.1',
    port='8812',
)
```

```python
index = "SPX"
# AND (Date BETWEEN '2009-12-31' AND '2029-12-31')
query = f"""
SELECT * 
FROM indexes
WHERE Symbol = '{index}'
"""

df = pd.read_sql(query, con=engine)
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
df = df.set_index("Date")
df.head()
```

# Returns and Volatility

```python
prices = df.AdjClose.values
df["log_ret"] = np.append(np.array(0.), np.log(prices[1:]/prices[:-1]))


window_len = 20
df["p_pos"] = df.log_ret.rolling(window_len).apply(lambda window: np.mean(window >= 0))
df["vol"] = df.log_ret.rolling(window_len).apply(lambda window: np.mean(window**2)**0.5)
df["vol_mean_med"] = df.log_ret.rolling(window_len).apply(lambda window: np.mean(window**2)/np.median(window**2))

df["vol_60d_trend"] = df.log_ret.rolling(60).apply(lambda window: get_robust_trend_coef(np.abs(window.values)))
df["vol_260d_trend"] = df.log_ret.rolling(260).apply(lambda window: get_robust_trend_coef(np.abs(window.values)))
df["vol_260d_range"] = (
    df.vol.rolling(260)
    .apply(lambda window: (window[-1] - window.min()) / (window.max() - window.min() + 1e-6))
)

fig, ax = plt.subplots(figsize=(15,5))
df["vol"].plot(ax=ax)
```

```python
vol = df["vol"].values
vol = vol[np.isfinite(vol)]
fig, ax = plt.subplots(figsize=(15,5))
_ = ax.hist(vol*(252**0.5), bins=100)
print(np.quantile(vol*(252**0.5), q=[0,0.05,0.25,0.5,0.75,0.95,1]))
```

```python
vol_lognorm = np.log(vol)
mu, std = vol_lognorm.mean(), vol_lognorm.std()
vol_lognorm = (vol_lognorm - mu) / std

fig, ax = plt.subplots(figsize=(15,5))
_ = ax.hist(vol_lognorm, bins=100)
```

```python
vol_dist_fxn = stats.powerlognorm
vol_dist_params = vol_dist_fxn.fit(data=vol_lognorm)
print(vol_dist_params)

fig, ax = plt.subplots(figsize=(10,10))

(osm, osr), (slope, intercept, r)= stats.probplot(
    vol_lognorm, 
    sparams=vol_dist_params,
    dist=vol_dist_fxn,
    plot=ax
)
print(slope, intercept, r)

vol_dist = vol_dist_fxn(*vol_dist_params)
samples = vol_dist.ppf(np.random.rand(len(vol_lognorm)))

fig, ax = plt.subplots(figsize=(15,5))
_, bins, _ = ax.hist(samples, bins=100, alpha=0.5, color="black")
_ = ax.hist(vol_lognorm, bins=bins, alpha=0.5, color="orange")

samples = np.exp(samples*std + mu)*(252**0.5)

fig, ax = plt.subplots(figsize=(15,5))
_, bins, _ = ax.hist(samples, bins=100, alpha=0.5, color="black")
_ = ax.hist(vol*(252**0.5), bins=bins, alpha=0.5, color="orange")
```

## Draw from a specific bin

```python
bin_number = 19

samples = vol_dist.ppf(bin_number*0.05 + np.random.rand(1000000)*0.05)
samples = np.exp(samples*std + mu)*(252**0.5)

fig, ax = plt.subplots(figsize=(15,5))
_, bins, _ = ax.hist(samples, bins=100, alpha=0.5, color="black")
```

# Volatility Bins & Statistical Signatures

```python
n_bins = 20
vol_bins = np.quantile(df.vol[np.isfinite(df.vol)], q=np.linspace(0, 1, n_bins + 1))
vol_bins = np.array(
    [
        0.0, 
        0.0039213 , 0.00456285, 0.00504972, 0.00544955,
        0.00584474, 0.00623728, 0.00659392, 0.0069622 , 
        0.00739167, 0.00784732, 0.00831799, 0.00884865, 
        0.00944347, 0.01021807, 0.01118019, 0.01240218, 
        0.0141092 , 0.01695328, 0.02268766,
        1.0
    ]
)

df["vol_bin"] = np.array([
    int(x) if np.isfinite(x) else None 
    for x in pd.cut(df.vol, bins=vol_bins, labels=False)
])
avg_bin_vol = df.loc[:,["vol_bin", "vol"]].groupby("vol_bin").mean()
len(avg_bin_vol)
```

```python
normalized_returns = defaultdict(list)

for i, window in enumerate(df.rolling(window_len + 1)):
    if len(window) == window_len + 1:
        vol_bin = int(window.vol_bin[-1])
        norm_rets = (window.log_ret / window.vol[-1]).values
        normalized_returns[vol_bin].append(
            pd.DataFrame({"norm_ret": norm_rets[1:], "prev_ret": norm_rets[:-1]})
        )
        
normalized_returns = [
    pd.concat(normalized_returns[vol_bin], axis=0) for vol_bin in range(n_bins)
]
```

```python
summary_stats = []
for i, rets in enumerate(normalized_returns):

    norm_rets = rets.norm_ret.values
    prev_rets = rets.prev_ret.values
    
    p = norm_rets[norm_rets >= 0]
    p = np.append(p, -p)
    _, p_norm_prob = stats.normaltest(p)
    p_df, p_loc, p_scale = stats.t.fit(data=p)
    p_kurt = stats.kurtosis(p)

    n = norm_rets[norm_rets < 0]
    n = np.append(n, -n)
    _, n_norm_prob = stats.normaltest(n)
    n_df, n_loc, n_scale = stats.t.fit(data=n)
    n_kurt = stats.kurtosis(n)
    
    summary_stats.append({
        "bin": i,
        "proba_up": (norm_rets >= 0).mean(),
        "proba_up_up": (norm_rets[prev_rets >= 0] >= 0).mean(),
        "proba_up_down": (norm_rets[prev_rets < 0] >= 0).mean(),
        "abs_auto_corr": np.corrcoef(np.abs(norm_rets), np.abs(prev_rets))[0,-1],
        "p_norm_prob": p_norm_prob,
        "p_df": p_df,
        "p_loc": p_loc, 
        "p_scale": p_scale,
        "p_kurt": p_kurt,
        "n_norm_prob": n_norm_prob,
        "n_df": n_df,
        "n_loc": n_loc, 
        "n_scale": n_scale,
        "n_kurt": n_kurt,
    })
    
summary_stats = DataFrame(summary_stats)
summary_stats["avg_vol"] = avg_bin_vol
```

```python
summary_stats.to_csv("2010-2021.csv", index=False)
```

# Bin Prediction Model

```python
X = df.loc[:, ["vol_bin", "p_pos","vol_mean_med", "vol_60d_trend", "vol_260d_trend", "vol_260d_range"]].values[600:-20]
y = df.vol_bin.values[620:].astype(int)
```

```python
model = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=50,
)

model.fit(X, y)
probas = model.predict_proba(X)
model.feature_importances_
```

```python
precision, recall, fscore, *_ = precision_recall_fscore_support(y, probas.argmax(-1))
print(f"   {np.mean(precision):>5.1%} {np.mean(recall):>5.1%} {np.mean(fscore):>5.1%}")
for i in range(20):
    print(f"{i:>2} {precision[i]:>5.1%} {recall[i]:>5.1%} {fscore[i]:>5.1%}")
```

```python
fig, ax = plt.subplots(figsize=(15,5))
ax.bar(x=np.arange(20), height=probas[y == 8].mean(0))
```

```python
fig, ax = plt.subplots(figsize=(15,5))
_ = ax.boxplot(
    probas[y == 9],
    sym='',                                 # No fliers
    whis=(5, 95),                           # 5% and 95% whiskers
    widths=0.8,
)
```

# Create sample distribution

```python
idx = 100
print(X[idx])
bin_transition_probas = probas[idx]
```

```python
seq_len = 20
n_samples = 10000

bin_samples = np.random.choice(np.arange(seq_len), size=n_samples, replace=True, p=bin_transition_probas)
vol_samples = np.exp(vol_dist.ppf(bin_samples*0.05 + np.random.rand(n_samples)*0.05)*std + mu)
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(vol_samples*(252**0.5), bins=30)
(vol_samples*(252**0.5)).mean()
```

```python
updown_seqs = []

rand_draws = np.random.rand(n_samples, seq_len + 1)
prev_states = np.where(rand_draws[:,0] <= summary_stats.proba_up[bin_samples], 1, -1)
for i in range(seq_len):
    thresholds = np.where(
        prev_states == 1, 
        summary_stats.proba_up_up[bin_samples], 
        summary_stats.proba_up_down[bin_samples]
    )
    updown_seqs.append(np.where(rand_draws[:, i + 1] <= thresholds, 1, -1))
    prev_states = updown_seqs[-1]
    
updown_seqs = np.stack(updown_seqs, axis=1)

norm_ret_samples = updown_seqs * np.abs(np.random.randn(n_samples, seq_len))
log_ret_samples = (vol_samples[:, None] * norm_ret_samples).sum(-1)
effective_vol = (log_ret_samples**2 * 252 / seq_len).mean()**0.5
effective_vol
```

```python
fig, ax = plt.subplots(figsize=(10,5))
_, bins, _ = ax.hist(np.exp(log_ret_samples) - 1, bins=100, alpha=0.5)

gbm_returns = sample_gbm_returns(
    drift=0, 
    volatility=effective_vol, 
    t=seq_len/252, 
    size=n_samples
)
_ = ax.hist(gbm_returns, bins=bins, alpha=0.5, color="orange")
```

```python

```

```python

```

```python
from sklearn.mixture import GaussianMixture
log_rets_centered = log_ret_samples - log_ret_samples.mean()
gmm = GaussianMixture(n_components = 7).fit(log_rets_centered[:, None])

fig, ax = plt.subplots(figsize=(15,10))
_, bins, _ = ax.hist(
    log_rets_centered, 
    bins=200, 
    density=True, 
    alpha=0.25, 
    label="Observed Distribution",
    color="black"
)

mixture_component_densities = []
for i, (wt, mean, cov) in enumerate(zip(gmm.weights_, gmm.means_, gmm.covariances_)):
    mixture_component_densities.append(wt*stats.norm.pdf(bins, mean, cov**0.5)[0])
    ax.plot(bins, mixture_component_densities[-1], label=f"Component {i}")
mixture_densities = np.stack(mixture_component_densities, -1).sum(-1)
ax.plot(bins, mixture_densities, color="black", label="Mixture")
ax.legend()


```

```python
from datetime import datetime
from typing import Dict, List, Union, Tuple
from numpy import ndarray


def get_mixture_price(
    option: Option,
    date: Union[datetime, str],
    spot: float,
    weights: ndarray,
    drifts: ndarray,
    vols: ndarray,
    dividend_yield: float = None,
) -> float:
    
    prices = np.array([
        get_bsm_price(
            option, date, spot, vol, drift, dividend_yield
        )
        for drift, vol in zip(drifts, vols)
    ])
    
    return np.sum(prices * weights)
```

```python
calls_bsm = []
puts_bsm = []
calls_mix = []
puts_mix = []
calls_ivs = []
put_ivs = []

expiration_date = "2022-05-31"
date = "2022-04-30"
spot = 100
rf_rate = 0.0
dvd_yld = 0.0
t = get_timediff(date, expiration_date)

weights, drifts, vols = gmm.weights_.flatten(), gmm.means_.flatten() / t, (gmm.covariances_.flatten() / t)**0.5

strikes = np.exp((effective_vol**2/252*seq_len)**0.5 * np.linspace(-3, 3, 100))*spot
for strike in strikes:
    result = get_statrep_prices(
        np.exp(log_ret_samples) - 1, t, strike, spot, rf_rate, dvd_yld
    )
    call_option = Option(ContractType.CALL, strike, expiration_date)
    calls_bsm.append(
        get_bsm_price(call_option, date, spot, effective_vol, rf_rate, dvd_yld)
    )
    calls_mix.append(
#         get_mixture_price(call_option, date, spot, weights, drifts, vols, dvd_yld)
        result.call
    )
    calls_ivs.append(
        get_bsm_iv(calls_mix[-1], call_option, date, spot, effective_vol, rf_rate, dvd_yld)
    )

    put_option = Option(ContractType.PUT, strike, expiration_date)
    puts_bsm.append(
        get_bsm_price(put_option, date, spot, effective_vol, rf_rate, dvd_yld)
    )
    puts_mix.append(
#         get_mixture_price(put_option, date, spot, weights, drifts, vols, dvd_yld)
        result.put
    )
    put_ivs.append(
        get_bsm_iv(puts_mix[-1], put_option, date, spot, effective_vol, rf_rate, dvd_yld)
    )
```

```python

```

```python
fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(strikes / spot, calls_ivs)
_ = ax.scatter(strikes / spot, put_ivs)
_ = ax.axhline(y=effective_vol)
```

```python
fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(strikes / spot, puts_bsm)
_ = ax.scatter(strikes / spot, puts_mix)

```

```python
from scipy.optimize import minimize

    
    return result.x[0]
```

```python

```
