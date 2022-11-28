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
import scipy.stats as stats

from numpy import ndarray
from quant_kit_core.distributions import get_best_fit_distribution
from quant_kit_core.time_series import (
    get_robust_trend_coef, 
    get_rolling_windows,
)


# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
```

```python
candidate_distributions = [
    "exponnorm",
    "genlogistic",
    "johnsonsu",
    "nct",
    "norm",
    "norminvgauss",
    "powerlognorm",
    "t",
]
```

# Load data

```python
df = pd.read_csv("../data/world-indices/$SPX.csv")
```

```python
df
```

# Rolling returns and vol

```python
prices = df.Close.values
log_rets = np.log(prices[1:]/prices[:-1])
```

## Boostrapped sample staistics

```python
n_boostraps = 1000
samples = np.random.choice(log_rets, size=(n_boostraps, len(log_rets)), replace=True)
means = samples.mean(axis=-1)
stds = (samples**2).mean(axis=-1)**0.5

# Normalization stats
mu, std = means.mean(), stds.mean()
norm_log_rets = (log_rets - mu) / std

fig, axs = plt.subplots(ncols=2, figsize=(15,7))
axs[0].hist(means*250, bins=30, edgecolor="black")
axs[1].hist(stds*250**0.5, bins=30, edgecolor="black")
```

```python
mean_dist = get_best_fit_distribution(
    means,
    candidate_distributions=candidate_distributions,
    support=(-np.inf, np.inf),
    n_restarts=100,
    fit_time_limit=30,
)[0]

mean_dist
```

```python
std_dist = get_best_fit_distribution(
    stds,
    candidate_distributions=candidate_distributions,
    support=(-np.inf, np.inf),
    n_restarts=100,
    fit_time_limit=30,
)[0]
std_dist
```

## Rolling windows

```python
window_size = 60
med_vol_window_size = window_size * 4 * 10

norm_log_ret_windows = get_rolling_windows(norm_log_rets, window_size)
print(norm_log_ret_windows.shape)

# Rolling stats
rolling_stats = dict(
    dates = df.Date.values[np.arange(window_size, len(df))],
    avgs = norm_log_ret_windows.mean(-1),
    vols = np.mean(norm_log_ret_windows**2, axis=-1)**0.5,
    trends = np.apply_along_axis(get_robust_trend_coef, axis=-1, arr=norm_log_ret_windows),
)

vol_median = np.median(get_rolling_windows(rolling_stats["vols"], med_vol_window_size), axis=-1)
rolling_stats["vol_median"] = np.concatenate((
    np.full(shape=(med_vol_window_size - 1,), fill_value=vol_median[0]),
    vol_median,
))
```

```python
fig, axs = plt.subplots(nrows=3, figsize=(15,15))
ticks = np.arange(0, len(rolling_stats["dates"]), 250*5)

axs[0].plot(rolling_stats["avgs"])
axs[0].axhline(y=0, color="gray")
axs[0].set_xticks(ticks)
axs[0].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[0].set_title("Rolling avg return")

axs[1].plot(rolling_stats["vols"])
axs[1].plot(rolling_stats["vol_median"], color="orange")
axs[1].axhline(y=np.median(rolling_stats["vols"]), color="gray")
axs[1].set_xticks(ticks)
axs[1].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[1].set_title("Rolling vol")

axs[2].plot(rolling_stats["trends"])
axs[2].set_xticks(ticks)
axs[2].set_xticklabels([dt[:4] for dt in rolling_stats["dates"][ticks]])
axs[2].set_title("Rolling trends")
```

# Environment segmentation

```python
step_size = window_size * 4 * 15
```

## Volatility environments

```python
vol_flags = np.where(rolling_stats["vols"] <= rolling_stats["vol_median"], 0, 1)
vol_states, vol_states_ct = np.unique(vol_flags, return_counts=True)
state_freqs = vol_states_ct / vol_states_ct.sum()


print([f"{s}:{f:>4.0%}" for s, f in zip(vol_states, state_freqs)])
print("-"*20)
for i in np.arange(0, len(vol_flags), step_size)[1:]:
    _, vol_states_ct = np.unique(vol_flags[i-step_size:i], return_counts=True)
    state_freqs = vol_states_ct / vol_states_ct.sum()
    print([f"{s}:{f:>4.0%}" for s, f in zip(vol_states, state_freqs)])
```

## Trend environments

```python
trend_flags = np.digitize(rolling_stats["trends"], bins=[-1.0, -0.25, 0.25, 1.0]) - 1
trend_states, trend_states_ct = np.unique(trend_flags, return_counts=True)
state_freqs = trend_states_ct / trend_states_ct.sum()

print([f"{s}:{f:>4.0%}" for s, f in zip(trend_states, state_freqs)])
print("-"*30)
for i in np.arange(0,len(trend_flags), step_size)[1:]:
    _, trend_states_ct = np.unique(trend_flags[i-step_size:i], return_counts=True)
    state_freqs = trend_states_ct / trend_states_ct.sum()
    print([f"{s}:{f:>4.0%}" for s, f in zip(trend_states, state_freqs)])
```

## Vol + Trend environments

```
0: Low  Vol | Downtrend
1: Low  Vol | No Trend
2: Low  Vol | Uptrend
3: High Vol | Downtrend
4: High Vol | No Trend
5: High Vol | Up Trend
```

```python
env_states_shape = (len(vol_states), len(trend_states))
env_states = np.arange(np.prod(env_states_shape)).reshape(env_states_shape)

env_flags = env_states[vol_flags, trend_flags]
print(env_flags.shape)

env_states, env_states_ct = np.unique(env_flags, return_counts=True)
state_freqs = env_states_ct / env_states_ct.sum()

print([f"{s}:{f:>4.0%}" for s, f in zip(env_states, state_freqs)])
print("-"*60)
for i in np.arange(0,len(env_flags), step_size)[1:]:
    _, env_states_ct = np.unique(env_flags[i-step_size:i], return_counts=True)
    state_freqs = env_states_ct / env_states_ct.sum()
    print([f"{s}:{f:>4.0%}" for s, f in zip(env_states, state_freqs)])
```

## Fit distribution to each env

```python
from quant_kit_core.distributions import ContinuousDistribution
```

```python
env_dists = dict()

fig, axs = plt.subplots(
    nrows=env_states.size, 
    figsize=(10,5*env_states.size), 
    sharex=True,
)


bins = np.linspace(*np.quantile(norm_log_rets, q=[0, 1]), 100)
for ax, state in zip(axs, env_states.flatten()):
    # Filter to daily, normalized log returns for state
    state_norm_log_rets = norm_log_ret_windows[env_flags == state].reshape(-1)
    
    # Fit distribution
    dist = ContinuousDistribution.from_name("norminvgauss").fit(state_norm_log_rets)
    env_dists[state] = dist
    
    # Print state stats
    print(f"{state}: {env_dists[state].name:<30} {state_norm_log_rets.mean():>10.4f} {state_norm_log_rets.std():>10.4f}")
    
    _, bins, _ = ax.hist(state_norm_log_rets, bins=bins, density=True, edgecolor="black")
    ax.plot(bins, env_dists[state].pdf(bins), color="black")
    ax.set_title(f"{state}")
```

# Transition matrix

```
        0   1   2   3   4   5
-------------------------------        
0 |  [[ 0,  1,  2,  3,  4,  5],
1 |   [ 6,  7,  8,  9, 10, 11],
2 |   [12, 13, 14, 15, 16, 17],
3 |   [18, 19, 20, 21, 22, 23],
4 |   [24, 25, 26, 27, 28, 29],
5 |   [30, 31, 32, 33, 34, 35]]
```

```python
n_states = env_states.size
transition_states = np.arange(n_states**2).reshape(n_states, n_states)
transition_states
```

## Sequences of non-overlapping environments

```python
n = len(env_flags)
n_remainder = n % window_size
start = n_remainder // 2
stop = n - (n_remainder - start)
n_splits = (stop - start) // window_size
non_overlapping_idxs = np.arange(start, stop)
non_overlapping_idxs = np.array(np.split(non_overlapping_idxs, n_splits))
non_overlapping_envs = env_flags[non_overlapping_idxs]
non_overlapping_envs.shape
```

## Rolling samplining windows to estimate transition matrix

```python
def get_transition_probas(
    non_overlapping_envs: ndarray, env_states: ndarray = None
) -> ndarray:
    """Estimate transition probabilities from a series of non-overlapping environment sequences
    
    Parameters
    ----------
    non_overlapping_envs: ndarray, shape(seq_len, n_non_overlapping)
        Sequences of non-overlapping environments (1 sequence per column)
    env_states: ndarray, shape=(n_envs,)
        All posible environment states.
        If ``None``, states will be inferred from sequences.
        (default=None)
        
    Returns
    -------
    transition_probas: ndarray, shape=(n_envs, n_envs)
        Probability of transitioning from a given environment to all other environments.
        Each row represents the transition probabilites for a given environment.
        Rows should sum to 1.
    """
    
    if env_states is None:
        env_states = np.unique(non_overlapping_envs)
    else:
        env_states = env_states[np.argsort(env_states)]
    
    # Code book for looking up a transition state given ``env_state_0`` and ``env_state_1``
    n_env_states = len(env_states)
    n_transition_states = n_env_states**2
    transition_states = np.arange(n_transition_states).reshape(n_env_states, n_env_states)
    
    # Code the sequences of environments by transition states
    env_states_0 = non_overlapping_envs[:-1]
    env_states_1 = non_overlapping_envs[1:]
    transitions = transition_states[env_states_0, env_states_1]
    
    # Count the number of times transitioned for each pair of environment states
    transition_counts = np.array(
        [
            (transitions == ts).sum() 
            for ts in transition_states.flatten()
        ]
    ).reshape(n_env_states, n_env_states).astype(np.float64)
    
    # Normalized by counts of each environment
    transition_probas = transition_counts / transition_counts.sum(-1, keepdims=True)
    
    return transition_probas
```

```python
sampling_window_size = int(250 /window_size * 10)

rws = get_rolling_windows(non_overlapping_envs, window_size=sampling_window_size)

transition_matrices = np.apply_along_axis(
    lambda x, *args: get_transition_probas(x.reshape(rws.shape[1:]), env_states),
    axis=-1,
    arr=rws.reshape(rws.shape[0], -1)
)
assert np.allclose(transition_matrices.sum(-1), 1.)

transition_matrices.shape
```

```python
np.median(transition_matrices, 0).round(2)
```

```python
get_transition_probas(non_overlapping_envs, env_states).round(2)
```

# Simulations

```python
from typing import Dict

def get_next_state(
    state: int,
    transition_matrix: ndarray,
    rnd: np.random.RandomState = None
) -> int:
    """Given previous state and a NxN transition matrix, choose next state
    
    Parameters
    ----------
    state: int, range=[0, N)
        Current state
    transition_matrix: ndarray, shape=(N,N), range=(0., 1.)
        NxN transition probability matrix: p(Next State | Current State)
        Rows sum to 1.
    rnd: np.random.RandomState, optional
        (default = None)
        
    Returns
    -------
    next_state: int, range=[0, N]
    """
    if rnd is None:
        rnd = np.random.RandomState()
    
    n_states = len(transition_matrix)
    states = np.arange(n_states)
    
    conditional_transition_probas = transition_matrix[state]
    
    return rnd.choice(states, size=1, p=conditional_transition_probas)[0]


def simulate_markov_chain(
    n_steps: int,
    transition_matrix: ndarray,
    init_state: int = None,
    rnd: np.random.RandomState = None
) -> ndarray:
    """Simulate sequential transition steps given a NxN transition matrix
    
    Parameters
    ----------
    n_steps: int
        Number of simulation steps
    transition_matrix: ndarray, shape=(N,N), range=(0., 1.)
        NxN transition probability matrix: p(Next State | Current State)
    init_state: int, optional
        Initial state to seed simulation.
    rnd: np.random.RandomState, optional
        (default = None)
        
    Returns
    -------
    chain: ndarray, shape=(n_steps + 1,)
        init_state + simulated sequence of states using transition_matrix.
    """
    if rnd is None:
        rnd = np.random.RandomState()
        
    n_states = len(transition_matrix)
    states = np.arange(n_states)
    
    if init_state is None:
        # Choose intial state
        state = rnd.choice(states, size=1)[0]
    else:
        assert init_state in states
        state = init_state
        
    chain = [state]
    for _ in range(n_steps):
        state = get_next_state(state, transition_matrix, rnd)
        chain.append(state)
        
    return np.array(chain)


def simulate_return_sequence(
    n_windows: int,
    window_size: int,
    transition_matrix: ndarray,
    env_dists: Dict[int, ContinuousDistribution],
    ret_mean: float,
    ret_std: float,
    rnd: np.random.RandomState = None
):
    """
    Parameters
    ----------
    n_windows: int
        Number of non-overlapping windows to simulate.
    window_size: int
        Size of each windows in # of trading days.
    transition_matrix: ndarray
        NxN transition probability matrix: p(Next State | Current State)
    env_dists: Dict[int, ContinuousDistribution]
        Continuous distribution fit to the observed log normalized daily returns of each environment state.
    ret_mean: float
        Daily return mean to use for denormalization.
    ret_std: float
        Daily return std to use for denormalization.
    rnd: np.random.RandomState = None
    
    Returns
    -------
    env_states: ndarray, shape=(n_windows + 1,)
        Initial environment state + simulated sequence of states using transition_matrix.
    sim_rets: ndarray, shape=(n_windows, window_size)
        Sampled daily returns for each environment window in simulated sequence.
    
    """
    if rnd is None:
        rnd = np.random.RandomState()
        
    # Simulate environment states
    env_states = simulate_markov_chain(n_windows, transition_matrix, rnd=rnd)
    
    # Sample log-nomralized daily returns for each environment
    sim_rets = np.stack(
        [
            dist.sample(size=(n_windows, window_size)) 
            for dist in env_dists.values()
        ],
        axis=-1
    )
    
    # Index to the tranistioned environment for each step
    sim_rets = np.take_along_axis(sim_rets, env_states[1:, None, None], axis=-1).squeeze(-1)
    
    # Denormalize returns
    sim_rets = sim_rets * ret_std + ret_mean
    
    # Log rets -> real returns
    sim_rets = np.exp(sim_rets) - 1
    
    return env_states, sim_rets
```

```python
# sample a transition matrix
n_t_matrices = len(transition_matrices)
t_matrices_idxs = np.arange(n_t_matrices)
transition_matrix = transition_matrices[np.random.choice(t_matrices_idxs)]

# Sample ret, std
ret_mean = np.random.choice(means, 1)[0]
ret_std = np.random.choice(stds, 1)[0]

# Simulate returns
env_states, sim_rets = simulate_return_sequence(
    n_windows=30*4,  # 30 years of data
    window_size=60,
    transition_matrix=transition_matrix,
    env_dists=env_dists,
    ret_mean=ret_mean,
    ret_std=ret_std
)

print(env_states.shape, sim_rets.shape)

fig, ax = plt.subplots(figsize=(15,7))
ax.plot(np.exp(np.cumsum(np.log(sim_rets.reshape(-1) + 1), axis=-1)))
```

```python

```

# Sizing optimization

```python
import torch
import torch.nn as nn
```

```python
exposures = nn.Parameter(
    data = torch.exp(torch.randn(len(env_dists)))
)
exposures
```

```python
drawdown_threshold = -0.33
scaled_returns = exposures[env_states[:-1], None] * torch.tensor(sim_rets, dtype=torch.float32)
scaled_returns = scaled_returns.reshape(-1,)
scaled_log_returns = torch.log(scaled_returns + 1)
equity_curve = torch.cumsum(scaled_log_returns, -1)
max_equities = torch.cummax(equity_curve, -1).values
max_drawdown = torch.min(equity_curve - max_equities)
loss = -(equity_curve[-1] + torch.clip(max_drawdown - np.log(1 + drawdown_threshold), max=0))
```

```python
equity_curve[-1]
```

```python

```
