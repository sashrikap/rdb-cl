"""Utility functions for optimization.

Includes:
    * Dictionary tools
        - zero_fill_dict
    * Functional tools

"""


import numpy as onp
import jax.numpy as np
import functools, itertools
from collections import OrderedDict
from functools import partial, reduce
from toolz.functoolz import juxt
from operator import mul, add

# ==================================================
# ================ Dictionary Tools ================
# ==================================================


def sort_dict_by_keys(dict_, keys):
    """Sort `dict_` into `OrderedDict()` based on `keys`.

    Example:
        >>> function({'b': 1, 'a':2}, keys=['a', 'b'])
        >>> # get OrderedDict({'a': 2, 'b': 1})

    Note:
        * `dict_` must contain all `keys`

    """
    d = OrderedDict()
    for k in keys:
        assert k in dict_
        d[k] = dict_[k]
    return d


def multiply_dict_by_keys(da, db):
    """Multiply key-value pairs.

    Usage:
        * Trajectory cost computation multiply_dict_by_keys(weights, features)

    Note:
        * `db` (features) must contain all `da` (weights) keys

    """
    newd = OrderedDict()
    for k, va in da.items():
        assert k in db.keys()
        newd[k] = va * db[k]
    return newd


def subtract_dict_by_keys(da, db):
    """Subtract key-value pairs.

    Usage:
        * Trajectory cost comparisons

    Note:
        * `db` (typically features) must contain all `da` (features) keys

    """
    newd = OrderedDict()
    for k, va in da.items():
        assert k in db.keys()
        newd[k] = va - db[k]
    return newd


"""
==================================================
================ Functional Tools ================
==================================================
"""


def compose(*functions):
    """Compose list of functions.

    To replace `toolz.functoolz.compose`, which causes error in jax>=0.1.50.

    Example:
        >>> f(g(h(x))) == compose(f, g, h)(x)

    """
    return functools.reduce(
        lambda f, g: lambda *args: f(g(*args)), functions, lambda x: x
    )


def sum_funcs(funcs):
    """Sum up multiple functions, used in reward computation, etc.

    Example:
        >>> fn = sum_funcs([sum, mul])
        >>> fn([1, 2]) = 5

    """
    add_op = partial(reduce, add)
    return compose(add_op, juxt(funcs))


def weigh_funcs_runtime(funcs_dict):
    """Weigh multiple functions by runtime weights.

    Functions and weights are organized by dict.

    Example:
        >>> cost_fn = weigh_funcs_runtime(funcs_dict)
        >>> cost = cost_fn(xs, weights)

    Note:
        * MUST ensure that `funcs_dict` and `weights_dict` contain matching
        keys in the same order (use sort_dict_by_keys). This function does
        not perform checking.

    """

    funcs_list = list(funcs_dict.values())
    funcs_keys = list(funcs_dict.keys())

    def func(*args, weights):
        """
        Args:
            weights (ndarray): Weights.numpy_array, (weights_dim, nbatch)

        Note:
            fn(*args): output (nbatch, )

        """
        assert isinstance(weights, list) or isinstance(weights, np.ndarray)
        assert len(funcs_list) == len(weights)
        output = 0.0
        for fn, w in zip(funcs_list, weights):
            val = fn(*args)
            assert (
                w.shape == val.shape
            ), f"Weight shape mismatches value shape: {w.shape}, {val.shape}"
            output += w * val
        return output

    return func


def merge_dict_funcs(funcs_dict):
    """Execute a dictionary of functions individually.

    """

    def func(*args):
        output = OrderedDict()
        for key, fn in funcs_dict.items():
            output[key] = fn(*args)
        return output

    return func


def chain_dict_funcs(outer_dict, inner_dict, mapping={}):
    """Chain dictionaries of functions.

    Example:
        >>> output[key] = outer[key](inner[key](data))

    Args:
        mapping (dict): if key is in outer_dict but not in inner_dict,
            find inner_dict[mapping[key]]
            useful for reusing functions from inner dict

    """
    output = OrderedDict()
    for key, outer in outer_dict.items():
        # if key in inner_dict.keys():
        if key not in inner_dict:
            assert key in mapping
            inner = inner_dict[mapping[key]]
        else:
            inner = inner_dict[key]
        output[key] = compose(outer, inner)
    return output


def concat_funcs(funcs, axis=-1):
    """Concatenate output values of list of functions into a list.

    Example:
        >>> [next_x1, next_x2] = f([x1, x2])

    Note:
        * Useful for Dynamics function

    """
    concat = partial(np.concatenate, axis=axis)
    return compose(concat, juxt(funcs))


def stack_funcs(funcs, axis=0):
    """Concatenate output values of list of functions into a list.

    Example:
        >>> [next_x1, next_x2] = f([x1, x2])
    Note:
        * Useful for Dynamics function

    """
    stack = partial(np.stack, axis=axis)
    return compose(stack, juxt(funcs))


def combine_funcs(funcs):
    """Append output values of list of functions into a list.

    Example:
        >>> [output1, output2] = f(data)

    """
    return compose(np.asarray, juxt(funcs))


def index_func(fn, idx_pair=(0, -1)):
    """Register a function with index pair.

    Return:
        function

    Example:
        >>> fn = make_func(np.sum, (1, 3))
        >>> fn([1, 2, 2, 3]) = 4

    """
    assert type(idx_pair) == tuple and len(idx_pair) == 2

    def _func(data, *kargs):
        return fn(np.array(data)[..., np.arange(*idx_pair)], *kargs)

    return _func


def or_funcs(funcs):
    """Logical or."""

    def _func(*args):
        return bool(sum([fn(*args) for fn in funcs]))

    return _func


def and_funcs(funcs):
    """Logical or."""

    def _func(*args):
        return bool(onp.prod([fn(*args) for fn in funcs]))

    return _func


def not_func(func):
    """Logical negate."""

    def _func(*args):
        return bool(not func(*args))

    return _func


def debug_print(data):
    print(data)
    return data
