"""Utility Functions for Inference.
"""
import numpy as onp
import jax.numpy as np


def stack_dict_values(dicts):
    lists = []
    for dict_ in dicts:
        lists.append(np.array(list(dict_.values())))
    return np.stack(lists)
