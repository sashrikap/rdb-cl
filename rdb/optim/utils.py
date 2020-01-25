"""
==================================================
================ Dictionary Tools ================
==================================================
"""

import jax.numpy as np
import functools, itertools
import numpy as onp
from collections import OrderedDict
from functools import partial, reduce
from toolz.functoolz import juxt
from operator import mul, add

"""
==================================================
================ Dictionary Tools ================
==================================================
"""


def zero_fill_dict(dict_, keys):
    """When `dict_` does not contain all of `keys`, zero-fill the missing elements.

    Userful when designer provides weight_dict that does not use all features.

    """
    out = {}
    for key in keys:
        if key not in dict_:
            out[key] = 0.0
        else:
            out[key] = dict_[key]
    return out


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
        # if k in dict_:
        d[k] = dict_[k]
    return d


def concate_dict_by_keys(dicts):
    """Stack key-value pairs for list of dictionaries

    Example:
        >>> function([{'a': [1], 'b': [2]}, {'a': [2], 'b': [3]}])
        >>> # get {'a': [[1], [2]], 'b': [[2], [3]]}

    Note:
        * Can be slow: 0.25s for 500 items, do not use in heavy iterations

    """
    if len(dicts) == 0:
        return {}
    else:
        # Crude check of keys
        keys = dicts[0].keys()
        out_dict = {}
        for key in keys:
            out_dict[key] = onp.array([d[key] for d in dicts])
        return out_dict


def unconcate_dict_by_keys(dict_):
    """Opposite of concate_dict_by_keys.

    Example:
        >>> input: {'a': np.array(10, 15)}
        >>> ouptut [{'a': np.array(15,)}, ] * 10

    """
    keys = list(dict_.keys())
    values = list(dict_.values())
    out_dicts = []
    for i in range(len(dict_[keys[0]])):
        out_dicts.append(dict(zip(keys, [v[i] for v in values])))
    return out_dicts


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


# def weigh_funcs(funcs_dict, weights_dict):
#     """Weigh multiple functions by predefined weights.

#     Functions & weights are organized by dict.

#     Note:
#         * MUST ensure that `funcs_dict` and `weights_dict` contain matching
#         keys, e.g. when users' weights omit certain features

#     """
#     fns = []
#     funcs_keys, weights_keys = list(funcs_dict.keys()), list(weights_dict.keys())
#     assert len(funcs_keys) == len(
#         weights_keys
#     ), "Must keep funcs and weights the same length"
#     for key in funcs_keys:
#         fns.append(compose(partial(mul, weights_dict[key]), funcs_dict[key]))
#     add_op = partial(reduce, add)
#     return compose(add_op, juxt(fns))


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
        assert isinstance(weights, list) or isinstance(weights, np.ndarray)
        assert len(funcs_list) == len(weights)
        output = 0.0
        for fn, w in zip(funcs_list, weights):
            output += w * fn(*args)
        return output

    return func


def weigh_funcs_dict(funcs_dict, weights_dict):
    """Weigh multiple functions by predefined weights.

    Notes:
        * Functions & weights are organized by dictionary.
        * funcs_dict is inclusive of weights_dict

    """
    fns = []
    for key, fn in funcs_dict.items():
        assert key in weights_dict
        w = weights_dict[key]
        fns.append(compose(partial(mul, w), fn))
    add_op = partial(reduce, add)
    return compose(add_op, juxt(fns))


def weigh_funcs_dict_runtime(funcs_dict):
    """Weigh multiple functions by runtime weights.

    Functions and weights are organized by dictionary.

    Example:
        >>> cost_fn = weigh_funcs_runtime(funcs_dict)
        >>> cost = cost_fn(xs, weights_dict)

    """

    def func(*args, weights):
        assert isinstance(weights, dict)
        output = 0.0
        for key, fn in funcs_dict.items():
            assert key in weights
            output += weights[key] * fn(*args)
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


def chain_dict_funcs(outer_dict, inner_dict):
    """Chain dictionaries of functions.

    Example:
        >>> output[key] = outer[key](inner[key](data))

    """
    output = OrderedDict()
    for key, outer in outer_dict.items():
        # if key in inner_dict.keys():
        assert key in inner_dict
        output[key] = compose(outer, inner_dict[key])
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
    stack = partial(np.stack, axis=0)
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


# def merge_list_funcs(funcs_list):
#     """Execute a list of functions individually.

#     """

#     def func(*args):
#         output = []
#         for fn in funcs_list:
#             output.append(fn(*args))
#         return output

#     return func


# def chain_list_funcs(outer_list, inner_list):
#     """Chain lists of functions.

#     Example:
#         >>> output[i] = outer[i](inner[i](data))

#     """
#     output = []
#     for outer, inner in zip(outer_list, inner_list):
#         output.append(compose(outer, inner))
#     return output
