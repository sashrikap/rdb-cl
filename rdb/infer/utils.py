"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np
import copy
import numpyro
import numpyro.distributions as dist
from rdb.optim.utils import concate_dict_by_keys
from numpyro.handlers import seed
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm, trange
from rdb.exps.utils import Profiler


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


def logsumexp(vs):
    max_v = onp.max(vs)
    ds = vs - max_v
    sum_exp = onp.exp(ds).sum()
    return max_v + onp.log(sum_exp)


def random_choice(items, num, probs=None, replacement=True):
    if not replacement:
        # no replacement
        assert probs is None, "Cannot use probs without replacement"
        assert num < len(items), f"Only has {len(items)} items"
        arr = numpyro.sample(
            "random_choice", dist.Uniform(0, 1), sample_shape=(len(items),)
        )
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
        arr = onp.repeat(arr[:, None], len(items), axis=1)
        diff = arr - probs
        idxs = onp.argmax(diff < 0, axis=1)
        output = [items[idx] for idx in idxs]
        return output


def random_uniform():
    return numpyro.sample("random", dist.Uniform(0, 1))


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


def prior_sample(log_prior_dict):
    """Sample prior distribution.

    Args:
        log_prior_dict (dict): maps keyword -> log probability

    Note:
        * log_prior_dict is LOG VALUE
        * Need seed(prior_sample, rng_key) to run

    """
    output = {}
    for key, dist_ in log_prior_dict.items():
        val = numpyro.sample(key, dist_)
        output[key] = onp.exp(val)
        # print(key, val)
    return output


def prior_log_prob(sample_dict, log_prior_dict):
    """Measure sample likelihood, based on prior.

    Args:
        sample_dict (dict): maps keyword -> value
        log_prior_dict (dict): maps keyword -> numpyro.dist

    Note:
        * Sample dict is RAW VALUE
          log_prior_dict is LOG VALUE
        * Currently only supports uniform distribution

    """

    def check_range(sample_val, prior_dist):
        """Check the range of sample_val against prior dist.

            Note:
            * numpyro.dist does not handle range in a "quiet" way
              e.g. `dist.Uniform(0, 1).log_prob(10)` will not give 0.
            * Let's fix this

        """
        assert isinstance(
            prior_dist, dist.Uniform
        ), f"Type `{type(prior_dist)}` supported"
        low = prior_dist.low
        high = prior_dist.high
        return onp.where(
            sample_val < low or sample_val > high,
            -onp.inf,
            prior_dist.log_prob(sample_val),
        )

    log_prob = 0.0
    for key, dist_ in log_prior_dict.items():
        val = sample_dict[key]
        log_val = onp.log(val)
        # print(f"{key} {log_val} range {check_range(log_val, dist_)}")
        log_prob += check_range(log_val, dist_)

    return log_prob


def gaussian_proposal(state, log_std_dict):
    """Propose next state given current state, based on Gaussian dist.

    Args:
        log_std_dict (dict): std of log(var)

    """
    next_state = copy.deepcopy(state)
    for key, val in next_state.items():
        if key in log_std_dict.keys():
            # Convert to log val
            log_val = onp.log(val)
            std = log_std_dict[key]
            next_log_val = numpyro.sample("next_log_val", dist.Normal(log_val, std))
            # Convert back to normal val
            next_state[key] = onp.exp(next_log_val)
    return next_state


def normalizer_sample(sample_fn, num):
    """Run sample function fixed number of times to generate samples.
    """
    samples = []
    for _ in range(num):
        # for _ in trange(num, desc="Normalizer weights"):
        samples.append(sample_fn())
    return samples
