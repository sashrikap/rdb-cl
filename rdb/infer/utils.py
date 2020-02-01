"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np
import jax
import copy, os
import numpyro
import numpyro.distributions as dist
from rdb.visualize.plot import plot_weights_comparison
from rdb.optim.utils import concate_dict_by_keys
from scipy.stats import gaussian_kde
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from numpyro.handlers import seed
from functools import partial

# ========================================================
# ============== Dictionary Tools ========================
# ========================================================


def stack_dict_values(dicts, normalize=False):
    """Stack a list of dictionaries into a list.

    Note:
        * Equivalent to stack([d.values for d in dicks])

    """
    lists = []
    for dict_ in dicts:
        lists.append(onp.array(list(dict_.values())))
    output = onp.stack(lists)
    if normalize:
        max_ = output.max(axis=0)
        min_ = output.min(axis=0)
        output = (output - min_) / (max_ - min_ + 1e-6)
    return output


def stack_dict_values_ratio(dicts, original):
    """Stack a list of dictionaries' ratio over original into a list.
    """
    lists = []
    for dict_ in dicts:
        lists.append(
            onp.array(list(dict_.values())) / onp.array(list(original.values()))
        )
    return onp.stack(lists)


def stack_dict_log_values(dicts):
    """Stack a list of log dictionaries into a list.

    """
    lists = []
    for dict_ in dicts:
        lists.append(onp.array(list(dict_.values())))
    output = onp.log(onp.stack(lists))
    return output


def select_from_dict(dict_, idx):
    """
    """
    output = {}
    for key, val in dict_.items():
        assert len(val) > idx
        output[key] = val[idx]
    return output


def random_choice_from_dict(dict_, random_choice_fn, num_samples, replacement=True):
    """
    """


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
    feats = concate_dict_by_keys(feats)
    feats_sum = concate_dict_by_keys(feats_sum)
    violations = concate_dict_by_keys(violations)
    return actions, feats, feats_sum, violations


# ========================================================
# ============== Visualization Tools =====================
# ========================================================


def visualize_chains(samples, accepts, num_plots, fig_dir, title, **kwargs):
    """Visualize multiple MCMC chains to check convergence.

    Args:
        samples (ndim=2): array of samples
        accepts (ndim=2, bool): array of acceptance
        fig_dir (str): directory to save the figure
        title (str): figure name, gets padded with plot index

    """
    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
    samples = onp.array(samples)
    accepts = onp.array(accepts)
    assert samples.shape[1] < len(colors), "Too many chains not enough colors"
    num_samples = len(samples) // num_plots
    os.makedirs(fig_dir, exist_ok=True)
    for i in range(num_plots):
        samples_i = samples[0 : num_samples * (i + 1), :]
        accepts_i = accepts[0 : num_samples * (i + 1), :]
        all_weights = []
        all_colors = []
        ratio = accepts_i[:, 0].sum() / accepts[:, 0].sum()
        title_i = f"{title}_accept_{100 * ratio:06.2f}%"
        path = f"{fig_dir}/{title_i}.png"
        for wi in range(samples.shape[1]):
            all_weights.append(samples_i[accepts_i[:, wi], wi])
            all_colors.append(colors[wi])
        plot_weights_comparison(
            all_weights, all_colors, path=path, title=title_i, **kwargs
        )
