"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np


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
