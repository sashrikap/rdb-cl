"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np
import copy
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import seed
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm


def stack_dict_values(dicts, normalize=False):
    """Stack a list of dictionaries into a list.

    Note:
        * Equivalent to stack([d.values for d in dicks])

    """
    lists = []
    for dict_ in dicts:
        lists.append(np.array(list(dict_.values())))
    output = np.stack(lists)
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
        lists.append(np.array(list(dict_.values())) / np.array(list(original.values())))
    return np.stack(lists)


def stack_dict_log_values(dicts):
    """Stack a list of log dictionaries into a list.

    """
    lists = []
    for dict_ in dicts:
        lists.append(np.array(list(dict_.values())))
    output = np.log(np.stack(lists))
    return output


def logsumexp(vs):
    max_v = np.max(vs)
    ds = vs - max_v
    sum_exp = np.exp(ds).sum()
    return max_v + np.log(sum_exp)


def random_choice(items, num):
    assert num < len(items), f"Only has {len(items)} items"
    arr = numpyro.sample("random_choice", dist.Uniform(0, 1), sample_shape=(num,))
    arr = np.argsort(arr)
    return [items[i] for i in arr]


def collect_features(self, list_ws, task, controller, runner, desc=None):
    """Utility for collecting features.

    Args:
        list_ws (list)
    """
    feats = []
    if desc is not None:
        list_ws = tqdm(list_ws, desc=desc)
    for w in list_ws:
        acs = controller(task, weights=w)
        _, _, info = runner(task, acs, weights=w)
        feats.append(info["feats"])
    return feats


def prior_sample(prior_dict):
    """Sample prior distribution.

    Args:
        prior_dict (dict): maps keyword -> numpyro.dist

    Note:
        * Need seed(prior_sample, rng_key) to run

    """
    output = {}
    for key, dist_ in prior_dict.items():
        output[key] = numpyro.sample(key, dist_)
    return output


def prior_log_prob(sample_dict, log_prior_dict):
    """Measure sample likelihood, based on prior.

    Args:
        sample_dict (dict): maps keyword -> value
        log_prior_dict (dict): maps keyword -> numpyro.dist

    Note:
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
        return np.where(sample_val < low or sample_val > high, -np.inf, 1.0)

    log_prob = 0.0
    for key, dist_ in log_prior_dict.items():
        val = sample_dict[key]
        log_prob += check_range(val, dist_) * dist_.log_prob(val)

    return log_prob


def gaussian_proposal(state, std_dict):
    next_state = copy.deepcopy(state)
    for key, val in next_state.items():
        log_val = np.log(val)
        if key in std_dict.keys():
            std = std_dict[key]
            next_log_val = numpyro.sample("next_log_val", dist.Normal(log_val, std))
            next_state[key] = np.exp(next_log_val)
    return next_state


def normalizer_sample(sample_fn, num):
    samples = []
    for _ in range(num):
        samples.append(sample_fn())
    return samples


def estimate_entropy(data, method="gaussian"):
    """
    Args:
        data (ndarray): (N, dim)
    """
    if method == "gaussian":
        # scipy gaussian kde requires transpose
        kernel = gaussian_kde(data.T)
        N = data.shape[0]
        entropy = -(1.0 / N) * np.sum(np.log(kernel(data.T)))
    elif method == "histogram":
        raise NotImplementedError
    return entropy
