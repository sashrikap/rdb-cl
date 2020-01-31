"""Test different optimizers in single & vectorized modes.

Include:
    [1] L-BFGS-BS

"""


import jax
import jax.numpy as np
from scipy.optimize import fmin_l_bfgs_b
from jax.config import config
from time import time
import numpy as onp

config.update("jax_enable_x64", True)


def numpy_fn(fn):
    def _fn(*args):
        return onp.array(fn(*args))

    return _fn


@jax.jit
def func_polynomial(x):
    x = np.array(x)
    x = np.reshape(x, (-1, 2))
    out = np.sum(
        8 * np.power(x[:, 0], 4) * np.power(x[:, 1], 4)
        + 7 * np.power(x[:, 0], 3) * np.power(x[:, 1], 3)
        + 6 * np.power(x[:, 0], 3) * np.power(x[:, 1], 2)
        + 5 * np.power(x[:, 0], 1) * np.power(x[:, 1], 1)
    )
    return out


@jax.jit
def rosen(x):
    """The Rosenbrock function"""
    x = np.array(x)
    x = np.reshape(x, (-1, 5))
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 + (1 - x[:, :-1]) ** 2.0)


def test_rosen_lbfgs():
    from scipy.optimize import minimize

    cost_fn = numpy_fn(rosen)
    grad_fn = numpy_fn(jax.jit(jax.grad(rosen)))
    data = onp.array([[1.3, 0.7, 0.8, 1.9, 1.2]])
    res = minimize(cost_fn, data, method="L-BFGS-B", jac=grad_fn)

    N = 1000
    data = onp.random.random((N, 5))
    _ = minimize(cost_fn, data, method="L-BFGS-B", jac=grad_fn)
    t1 = time()
    batch_res = minimize(cost_fn, data, method="L-BFGS-B", jac=grad_fn)
    tbatch = time() - t1
    batch_x = batch_res["x"]
    print(f"Batch time {tbatch:.3f}")

    single_x = []
    _ = minimize(cost_fn, data[0, :], method="L-BFGS-B", jac=grad_fn)
    t1 = time()
    for i in range(len(data)):
        single_res = minimize(cost_fn, data[i, :], method="L-BFGS-B", jac=grad_fn)
        single_x.append(single_res["x"])
    tsingle = time() - t1
    diffs = []
    for sx, bx in zip(single_x, batch_x):
        diffs.append(onp.abs(sx - bx).sum())
    print(f"Mean diff {onp.mean(diffs):.3f}")
    print(
        f"Batch time {tbatch:.3f} single time {tsingle:.3f} speed up {tsingle / tbatch:.2f}"
    )
    # 200: 45x, 500: 63x, 1000: 120x
