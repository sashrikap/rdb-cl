"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np
import jax
import copy, os
import numpyro
import numpyro.distributions as dist
from rdb.visualize.plot import plot_weights_comparison
from rdb.optim.utils import *
from rdb.infer.dictlist import *
from scipy.stats import gaussian_kde
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from numpyro.handlers import seed
from functools import partial


# ========================================================
# ============ Sampling & Numerical Tools ================
# ========================================================


def logsumexp(vs, axis=-1):
    max_v = onp.max(vs, axis=axis)
    ds = vs - onp.max(vs, axis=axis, keepdims=True)
    sum_exp = onp.exp(ds).sum(axis=axis)
    return max_v + onp.log(sum_exp)


def random_choice(items, num, probs=None, replacement=True):
    """Randomly sample from items.

    Usage:
        * Select all: num=-1
        * Sample without replacement: must satisfy num <= len(items)
        * Sample with replacement

    """
    if num < 0:
        return items

    if not replacement:
        # no replacement
        assert probs is None, "Cannot use probs without replacement"
        assert num < len(items), f"Only has {len(items)} items"
        probs = onp.array(probs)
        arr = numpyro.sample(
            "random_choice", dist.Uniform(0, 1), sample_shape=(len(items),)
        )
        arr = onp.array(arr)
        idxs = onp.argsort(arr)[:num]
        return [items[idx] for idx in idxs]
    else:
        # with replacement
        if probs is None:
            probs = onp.ones(len(items)) / len(items)
        else:
            assert len(probs) == len(items)
            probs = probs / onp.sum(probs)
        probs = onp.cumsum(probs)
        arr = numpyro.sample("random_choice", dist.Uniform(0, 1), sample_shape=(num,))
        arr = onp.array(arr)
        output = []
        for i in range(num):
            diff = onp.minimum(arr[i] - probs, 0)
            first_neg = onp.argmax(diff < 0)
            output.append(items[int(first_neg)])
        return onp.array(output)


def random_uniform(low=0.0, high=1.0):
    return numpyro.sample("random", dist.Uniform(low, high))


# ========================================================
# ================= Rollout Tools ========================
# ========================================================


def collect_trajs(list_ws, state, controller, runner, desc=None):
    """Utility for collecting features.

    Args:
        list_ws (DictList): (nweights, nbatch)
        state (ndarray): initial state for the task (1, xdim)

    Output:
        actions (ndarray): (nbatch, T, udim)
        feats (DictList): nfeats * (nbatch, T)
        feats_sum (DictList): nfeats * (nbatch,)
        violations (DictList): nvios * (nbatch, T)

    """
    feats = []
    feats_sum = []
    violations = []
    actions = []
    num_ws = len(list_ws)
    assert state.shape[0] == 1 and len(state.shape) == 2
    batch_states = onp.repeat(state, num_ws, axis=0)
    batch_ws = DictList(list_ws)

    ## acs (nbatch, T, udim)
    actions = controller(batch_states, weights=batch_ws)
    ## xs (T, nbatch, xdim), costs (nbatch)
    xs, costs, info = runner(batch_states, actions, weights=batch_ws)
    return actions, info["feats"], info["feats_sum"], info["violations"]


# ========================================================
# ============== Visualization Tools =====================
# ========================================================


def visualize_chains(chains, rates, fig_dir, title, **kwargs):
    """Visualize multiple MCMC chains to check convergence.

    Args:
        chains (list): list of accepted chains, [(n_samples, n_weights), ...]
        num_plots=10, visualize ~10%, 20%...99% samples
        fig_dir (str): directory to save the figure
        title (str): figure name, gets padded with plot index

    """
    import itertools
    import matplotlib.cm as cm

    assert len(chains) == len(rates)
    colors = cm.Spectral(np.linspace(0, 1, len(chains)))
    os.makedirs(fig_dir, exist_ok=True)
    all_weights = []
    all_colors = []
    all_labels = []
    for ci, (chain_i, rate_i) in enumerate(zip(chains, rates)):
        all_labels.append(
            f"Chain({ci}): accept {len(chain_i)} ({(100 * rate_i):.02f}%)"
        )
    plot_weights_comparison(
        chains, colors, all_labels, path=f"{fig_dir}/{title}.png", title=title, **kwargs
    )
