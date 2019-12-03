import jax.numpy as np
import functools
from collections import OrderedDict
from functools import partial, reduce
from toolz.functoolz import juxt
from operator import mul, add


def compose(*functions):
    """Compose list of functions

    Example:
    >>> f(g(h(x))) == compose(f, g, h)(x)

    """
    return functools.reduce(
        lambda f, g: lambda *args: f(g(*args)), functions, lambda x: x
    )


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


def weigh_funcs_runtime(funcs_dict):
    """
    Weigh multiple functions by runtime weights
    """

    def func(*args, weights_dict):
        output = 0.0
        for key, fn in funcs_dict.items():
            if key in weights_dict.keys():
                w = weights_dict[key]
                # w * fn(args)
                output += w * fn(*args)
        return output

    return func


def merge_dict_funcs(funcs_dict):
    """
    Execute a dictionary of functions individually
    """

    def func(*args):
        output = OrderedDict()
        for key, fn in funcs_dict.items():
            output[key] = fn(*args)
        return output

    return func


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


def concat_funcs(funcs, axis=-1):
    """
    Concatenate output values of list of functions into a list

    Useful for:
    [1] Dynamics function [next_x1, next_x2] = f([x1, x2])
    """
    concat = partial(np.concatenate, axis=axis)
    return compose(concat, juxt(funcs))


def stack_funcs(funcs, axis=0):
    """
    Concatenate output values of list of functions into a list

    Useful for:
    [1] Dynamics function [next_x1, next_x2] = f([x1, x2])
    """
    stack = partial(np.stack, axis=0)
    return compose(stack, juxt(funcs))


def combine_funcs(funcs):
    """
    Append output values of list of functions into a list

    Usage: `[output1, output2] = f(data)`
    """
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
