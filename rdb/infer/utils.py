"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np
import jax
import math
import copy, os
import numpyro
import numpyro.distributions as dist
from jax import random
from functools import partial
from rdb.infer.mcmc import MH
from rdb.optim.utils import *
from rdb.infer.dictlist import *
from numpyro.handlers import seed
from tqdm.auto import tqdm, trange
from rdb.exps.utils import Profiler
from scipy.stats import gaussian_kde
from numpyro.infer import HMC, MCMC, NUTS, SA
from rdb.visualize.plot import plot_weights_comparison

# from rdb.infer.mcmc import MH as RDB_MH

# ========================================================
# ============= Sampling Interface Tools =================
# ========================================================


def get_ird_sampler(
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
        kernel = MH(model, **init_args)
    elif name == "HMC":
        kernel = HMC(model, **init_args)
    elif name == "NUTS":
        kernel = NUTS(model, **init_args)
    elif name == "SA":
        kernel = SA(model, **init_args)
    else:
        raise NotImplementedError
    # return MCMC(kernel, progress_bar=True, **sampler_args)
    return MCMC(kernel, progress_bar=True, **sampler_args)


def get_designer_sampler(
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
        kernel = MH(model, jit=False, **init_args)
    else:
        raise NotImplementedError
    return MCMC(
        kernel,
        progress_bar=True,
        jit_model=False,
        chain_method="sequential",
        **sampler_args,
    )


# ========================================================
# ============ Sampling & Numerical Tools ================
# ========================================================


def random_choice(rng_key, items, num, probs=None, replacement=True):
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
        arr = random.uniform(rng_key, shape=(len(items),))
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
        arr = random.uniform(rng_key, shape=(num,))
        arr = onp.array(arr)
        output = []
        for i in range(num):
            diff = onp.minimum(arr[i] - probs, 0)
            first_neg = onp.argmax(diff < 0)
            output.append(items[int(first_neg)])
        return onp.array(output)


def random_uniform(rng_key, low=0.0, high=1.0):
    return random.uniform(rng_key, minval=low, maxval=high)


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


def collect_trajs(
    weights_arr,
    states,
    controller,
    runner,
    us0=None,
    desc=None,
    jax=False,
    max_batch=-1,
):
    """Utility for collecting features.

    Args:
        weights_arr (ndarray): (nfeats, nbatch)
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
    feats = None
    feats_sum = None
    violations = None
    actions = None
    xdim = states.shape[1]
    nfeats = weights_arr.shape[0]
    nbatch = weights_arr.shape[1]
    assert len(states.shape) == 2 and len(states) == nbatch
    if us0 is not None:
        if jax:
            us0 = np.array(us0)
        else:
            us0 = onp.array(us0)
        assert len(us0.shape) == 3 and len(us0) == nbatch

    if max_batch == -1:
        ## acs (nbatch, T, udim)
        actions = controller(
            states, us0=us0, weights=None, weights_arr=weights_arr, jax=jax
        )
        ## xs (T, nbatch, xdim), costs (nbatch)
        xs, costs, info = runner(
            states, actions, weights=None, weights_arr=weights_arr, jax=jax
        )
        feats, feats_sum, violations = (
            info["feats"],
            info["feats_sum"],
            info["violations"],
        )

    else:
        num_iterations = math.ceil(nbatch / max_batch)
        for it in range(num_iterations):
            it_begin = it * max_batch
            it_end = min((it + 1) * max_batch, nbatch)
            valid_i = onp.ones(max_batch, dtype=bool)
            idxs = onp.zeros(nbatch, dtype=bool)
            idxs[it_begin:it_end] = True
            states_i = states[list(idxs)]
            us0_i = None if not us0 else us0[list(idxs)]
            weights_arr_i = weights_arr[:, list(idxs)]
            if it == num_iterations - 1:
                # Pad state/weight pairs with 0 (valid=False)
                valid_i[it_end - it_begin :] = False
                num_pad = num_iterations * max_batch - nbatch
                states_pad = np.zeros((num_pad, xdim))
                states_i = np.concatenate([states_i, states_pad], axis=0)
                weights_pad = np.ones((nfeats, num_pad))
                weights_arr_i = np.concatenate([weights_arr_i, weights_pad], axis=1)
                if us0_i:
                    us0_pad = np.zeros((num_pad, us0_i.shape[1]))
                    us0_i = np.concatenate([us0_i, us0_pad], axis=0)
            actions_i = controller(
                states_i, us0=us0_i, weights=None, weights_arr=weights_arr_i, jax=jax
            )
            ## xs (T, nbatch, xdim), costs (nbatch)
            xs_i, costs_i, info_i = runner(
                states_i, actions_i, weights=None, weights_arr=weights_arr_i, jax=jax
            )
            feats_i, feats_sum_i = info_i["feats"], info_i["feats_sum"]
            violations_i = info_i["violations"]
            if it == 0:
                actions = actions_i[list(valid_i)]
                costs = costs_i[list(valid_i)]
                feats = feats_i[list(valid_i)]
                feats_sum = feats_sum_i[list(valid_i)]
                violations = violations_i[list(valid_i)]
            else:
                actions = np.concatenate([actions, actions_i[list(valid_i)]], axis=0)
                costs = np.concatenate([costs, costs_i[list(valid_i)]], axis=0)
                feats = feats.concat(feats_i[list(valid_i)], axis=0)
                feats_sum = feats_sum.concat(feats_sum_i[list(valid_i)], axis=0)
                violations = violations.concat(violations_i[list(valid_i)], axis=0)

    return actions, costs, feats, feats_sum, violations


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
