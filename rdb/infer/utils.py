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
from numpyro.infer import HMC, MCMC, NUTS, SA
from rdb.infer.mcmc import NumPyro_MH, RDB_MH

# from rdb.infer.mcmc import MH as RDB_MH

# ========================================================
# ============= Sampling Interface Tools =================
# ========================================================


def get_numpyro_sampler(
    name, model, init_args={}, sampler_args={"num_warmup": 10, "num_sample": 10}
):
    """
    Build numpyro kernel.

    Args:
        name (str)
        model (fn): designer/IRD model
        init_args (dict): initialization args

    """
    assert name in ["MH", "HMC", "NUTS", "SA"]
    if name == "MH":
        kernel = NumPyro_MH(model, **init_args)
    elif name == "HMC":
        kernel = HMC(model, **init_args)
    elif name == "NUTS":
        kernel = NUTS(model, **init_args)
    elif name == "SA":
        kernel = SA(model, **init_args)
    else:
        raise NotImplementedError
    return MCMC(kernel, progress_bar=True, **sampler_args)


def get_rdb_sampler(
    name, model, init_args={}, sampler_args={"num_warmup": 10, "num_sample": 10}
):
    """
    Build rdb kernel, which shares API as numpyro but avoids jitting the model.
    Currently only Metropolis Hasting implemented.

    Args:
        name (str)
        model (fn): designer/IRD model
        init_args (dict): initialization args

    """
    assert name in ["MH"]
    if name == "MH":
        kernel = RDB_MH(model, **init_args)
    else:
        raise NotImplementedError
    return MCMC(kernel, progress_bar=True, **sampler_args)


# ========================================================
# ============ Sampling & Numerical Tools ================
# ========================================================


def logsumexp(vs, axis=-1):
    print("Inside logsumexp", type(vs))
    max_v = onp.max(vs, axis=axis)
    ds = vs - onp.max(vs, axis=axis, keepdims=True)
    sum_exp = onp.exp(ds).sum(axis=axis)
    return max_v + onp.log(sum_exp)


def np_logsumexp(vs, axis=-1):
    max_v = np.max(vs, axis=axis)
    ds = vs - np.max(vs, axis=axis, keepdims=True)
    sum_exp = np.exp(ds).sum(axis=axis)
    return max_v + np.log(sum_exp)


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


def cross_product(data_a, data_b, type_a, type_b):
    """To do compuation on each a for each b.

    Performs Cross product: (num_a,) x (num_b,) -> (num_a * num_b,).

    Output:
        batch_a (type_a): shape (num_a * num_b,)
        batch_b (type_b): shape (num_a * num_b,)

    """

    pairs = list(itertools.product(data_a, data_b))
    batch_a, batch_b = zip(*pairs)
    return type_a(batch_a), type_b(batch_b)


def collect_trajs(ws, states, controller, runner, us0=None, desc=None):
    """Utility for collecting features.

    Args:
        ws (DictList): nfeats * (nbatch)
        states (ndarray): initial state for the task
            shape (nbatch, xdim)
        us0 (ndarray) initial action
            shape (nbatch, T, udim)

    Output:
        actions (ndarray): (nbatch, T, udim)
        costs (ndarray): (nbatch,)
        feats (DictList): nfeats * (nbatch, T)
        feats_sum (DictList): nfeats * (nbatch,)
        violations (DictList): nvios * (nbatch, T)

    """
    feats = []
    feats_sum = []
    violations = []
    actions = []
    num_ws = len(ws)
    assert isinstance(ws, DictList)
    assert len(states.shape) == 2 and len(states) == len(ws)
    assert len(ws.shape) == 1
    if us0 is not None:
        us0 = onp.array(us0)
        assert len(us0.shape) == 3 and len(us0) == len(ws)
    ## acs (nbatch, T, udim)
    actions = controller(states, us0=us0, weights=ws)
    ## xs (T, nbatch, xdim), costs (nbatch)
    xs, costs, info = runner(states, actions, weights=ws)
    return actions, costs, info["feats"], info["feats_sum"], info["violations"]


# ========================================================
# ============== Visualization Tools =====================
# ========================================================


def visualize_chains(chains, rates, fig_dir, title, **kwargs):
    """Visualize multiple MCMC chains to check convergence.

    Args:
        chains (DictList): list of accepted chains
            shape: nfeats * (nchains, nsamples)
        rates (list): acceptance rates
            shape: (nchains,)
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
