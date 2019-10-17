import jax.numpy as np
from collections import OrderedDict
from functools import partial, reduce
from toolz.functoolz import juxt, compose
from operator import mul, add


def sum_funcs(funcs):
    """
    Sum up multiple functions, used in reward computation, etc

    Usage:
    ```
    fn = sum_funcs([sum, mul])
    fn([1, 2]) = 5
    ```
    """
    add_op = partial(reduce, add)
    return compose(add_op, juxt(funcs))


def weigh_funcs(funcs_dict, weights_dict):
    """
    Weigh multiple functions by predefined weights
    """
    # pairs = [(fn, weights_dict[key]) for key, fn in funcs_dict.items()]
    fns = []
    for key, fn in funcs_dict.items():
        if key in weights_dict.keys():
            w = weights_dict[key]
            # w * fn(args)
            fns.append(compose(partial(mul, w), fn))

    add_op = partial(reduce, add)
    return compose(add_op, juxt(fns))


def chain_funcs(outer_dict, inner_dict):
    """
    Chain dictionaries of functions

    output[key] = outer[key](inner[key](data))
    """
    output = OrderedDict()
    for key, outer in outer_dict.items():
        if key in inner_dict.keys():
            output[key] = compose(outer, inner_dict[key])
    return output


def concat_funcs(funcs):
    return compose(np.concatenate, juxt(funcs))


def combine_funcs(funcs):
    return compose(np.asarray, juxt(funcs))


def index_func(fn, idx_pair=(0, -1)):
    """
    Register a function with index pair

    Usage:
    ```
    fn = make_func(np.sum, (1, 3))
    fn([1, 2, 2, 3]) = 4
    ```
    """
    assert type(idx_pair) == tuple and len(idx_pair) == 2

    def func(data, *kargs):
        return fn(np.array(data)[..., np.arange(*idx_pair)], *kargs)

    return func


def debug_print(data):
    print(data)
    return data
