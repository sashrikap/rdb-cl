from rdb.optim.utils import *
import jax.numpy as np


def test_sum():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    fn = sum_funcs([f1, f2])
    assert np.allclose(fn(1, 2), 2)


def test_weights():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    fns = {"f1": f1, "f2": f2}
    weights = {"f1": 1, "f2": 2}
    fn = weigh_funcs(fns, weights_dict=weights)
    assert np.allclose(fn(1, 2), 1)


def test_partial():
    from functools import partial

    def func(a, b, c):
        print(f"a {a} b {b} c {c}")

    p_func = partial(func, b=1)
    p_func(a=2, c=3)


def test_chain():
    def f1(a, b):
        return a + b

    def f2(a):
        return a ** 2

    inner = {"a": f1}
    outer = {"a": f2}
    chain = chain_funcs(outer, inner)
    assert np.allclose(chain["a"](1, 2), [9.0])


def test_index():
    fn = index_func(np.sum, (1, 3))
    assert np.allclose(fn([1, 2, 2, 3]), 4)
