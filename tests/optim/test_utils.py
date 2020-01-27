from rdb.optim.utils import *
from time import time
import jax.numpy as np
import jax
import numpy as onp


def test_sum():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    fn = sum_funcs([f1, f2])
    assert np.allclose(fn(1, 2), 2)


def test_append_dict():
    d1 = {"a": [1, 2, 3]}
    d2 = {"a": 4}
    out = {"a": [1, 2, 3, 4]}
    dout = append_dict_by_keys(d1, d2)
    for k, v in dout.items():
        assert onp.allclose(dout[k], out[k])


def test_weights_dict():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    def f3(a, b):
        return np.array(a) * np.array(b)

    fns = {"f1": f1, "f2": f2, "f3": f3}
    weights = {"f1": 1, "f2": 2}
    fn_keys = list(fns.keys())
    weights = zero_fill_dict(weights, fn_keys)
    fn = weigh_funcs_dict(fns, weights_dict=weights)
    assert np.allclose(fn(1, 2), 1)


def test_weights_dict_jax():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    def f3(a, b):
        return np.array(a) * np.array(b)

    fns = {"f1": f1, "f2": f2, "f3": f3}
    weights = {"f1": 1, "f2": 2}
    fn_keys = list(fns.keys())
    weights = sort_dict_by_keys(zero_fill_dict(weights, fn_keys), fn_keys)
    t1 = time()
    fn = jax.jit(weigh_funcs_dict_runtime(fns))
    assert np.allclose(fn(1, 2, weights=weights), 1)
    dt1 = time() - t1
    # print(f"Took time {dt1}")

    t2 = time()
    assert np.allclose(fn(1, 2, weights=weights), 1)
    dt2 = time() - t2
    # print(f"Took time {dt2}")

    weights = {"f1": 2, "f2": 2}
    fn_keys = list(fns.keys())
    weights = sort_dict_by_keys(zero_fill_dict(weights, fn_keys), fn_keys)
    t3 = time()
    assert np.allclose(fn(1, 2, weights=weights), 4)
    dt3 = time() - t3
    # print(f"Took time {dt3}")
    print(f"Dict version slow down {dt3/dt2:.2f}")


def test_weights_list_jax():
    def f1(a, b):
        return np.array(a) + np.array(b)

    def f2(a, b):
        return np.array(a) - np.array(b)

    fns = {"f1": f1, "f2": f2}
    weights = [1, 2]
    t1 = time()
    fn = jax.jit(weigh_funcs_runtime(fns))
    assert np.allclose(fn(1, 2, weights=weights), 1)
    dt1 = time() - t1

    t2 = time()
    assert np.allclose(fn(1, 2, weights=weights), 1)
    dt2 = time() - t2

    weights = [2, 2]
    t3 = time()
    assert np.allclose(fn(1, 2, weights=weights), 4)
    dt3 = time() - t3
    print(f"List version slow down {dt3/dt2:.2f}")


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
    chain = chain_dict_funcs(outer, inner)
    assert np.allclose(chain["a"](1, 2), [9.0])


def test_index():
    fn = index_func(np.sum, (1, 3))
    assert np.allclose(fn([1, 2, 2, 3]), 4)


def test_concate_dict_speed():
    from rdb.exps.utils import Profiler

    keys = ["a", "b", "c", "d", "e"]
    dicts = []
    for _ in range(500):
        dicts.append(dict(zip(keys, onp.random.random(5))))
    for _ in range(10):
        with Profiler("Concate dict"):
            concate_dict_by_keys(dicts)


def test_concat_funcs():
    def f1(x1, x2):
        return x1 - x2

    def f2(x1, x2):
        return x1 + 2 * x2

    concat_fn = concat_funcs([f1, f2], axis=-1)
    stack_fn = stack_funcs([f1, f2], axis=-1)
    x1 = np.array([1, 2])
    x2 = np.array([2, 3])
    stack_output = np.array([[-1, -1], [5, 8]])
    concat_output = np.array([-1, -1, 5, 8])
    assert np.allclose(stack_fn(x1, x2), stack_output)
    assert np.allclose(concat_fn(x1, x2), concat_output)


def test_concat_funcs_2d():
    def f1(x1, x2):
        return x1 - x2

    def f2(x1, x2):
        return x1 + 2 * x2

    concat_fn = concat_funcs([f1, f2], axis=-1)
    stack_fn = stack_funcs([f1, f2], axis=-1)
    x1 = np.array([[1, 2], [1, 2]])
    x2 = np.array([[2, 3], [2, 3]])
    stack_output = np.array([[[-1, -1], [-1, -1]], [[5, 8], [5, 8]]])
    concat_output = np.array([[-1, -1, 5, 8], [-1, -1, 5, 8]])
    assert np.allclose(stack_fn(x1, x2), stack_output)
    assert np.allclose(concat_fn(x1, x2), concat_output)
