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
from scipy.stats import gaussian_kde
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from numpyro.handlers import seed
from functools import partial


# ========================================================
# ============ Sampling & Numerical Tools ================
# ========================================================


def logsumexp(vs):
    max_v = onp.max(vs)
    ds = vs - max_v
    sum_exp = onp.exp(ds).sum()
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
            output.append(items[first_neg])
        return onp.array(output)


def random_uniform():
    return numpyro.sample("random", dist.Uniform(0, 1))


# ========================================================
# ================= Rollout Tools ========================
# ========================================================


def collect_trajs(list_ws, state, controller, runner, desc=None):
    """Utility for collecting features.

    Args:
        list_ws (list)
        state (ndarray): initial state for the task
    """
    feats = []
    feats_sum = []
    violations = []
    actions = []
    if desc is not None:
        list_ws = tqdm(list_ws, desc=desc)
    for w in list_ws:
        acs = controller(state, weights=w)
        _, _, info = runner(state, acs, weights=w)
        actions.append(acs)
        feats.append(info["feats"])
        feats_sum.append(info["feats_sum"])
        violations.append(info["violations"])
    feats = stack_dict_by_keys(feats)
    feats_sum = stack_dict_by_keys(feats_sum)
    violations = stack_dict_by_keys(violations)
    return actions, feats, feats_sum, violations


# ========================================================
# ============== Visualization Tools =====================
# ========================================================


def visualize_chains(chains, fig_dir, title, **kwargs):
    """Visualize multiple MCMC chains to check convergence.

    Args:
        chains (list): list of accepted chains, [(n_samples, n_weights), ...]
        num_plots=10, visualize ~10%, 20%...99% samples
        fig_dir (str): directory to save the figure
        title (str): figure name, gets padded with plot index

    """
    import itertools
    import matplotlib.cm as cm

    colors = cm.Spectral(np.linspace(0, 1, len(chains)))
    os.makedirs(fig_dir, exist_ok=True)
    all_weights = []
    all_colors = []
    all_labels = []
    for ci, chain_i in enumerate(chains):
        all_labels.append(f"Label_{ci}_accept_{len(chain_i)}")
    plot_weights_comparison(
        chains, colors, all_labels, path=f"{fig_dir}/{title}.png", title=title, **kwargs
    )
