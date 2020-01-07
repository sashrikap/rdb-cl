from jax.lax import fori_loop, scan
from jax.ops import index_update
from jax import jit
from time import time
import jax.numpy as np
import jax


def test_fori():
    def rollout(x, us, A, B):
        N = len(us)
        outs = np.zeros((len(us), len(x)))

        def step(t, carry):
            outs, last_x = carry
            u = us[t]
            next_x = A @ last_x + B @ u
            outs = index_update(outs, t, next_x)
            return outs, next_x

        fori_loop(0, N, step, (outs, x))
        return np.sum(np.square(outs))

    A = np.array([[7.0, 0.5], [0.5, 0.8]])
    B = np.array([[0.4, 1.2]])
    x = np.ones(2)
    us = np.ones((10, 2))
    out = rollout(x, us, A, B)

    # grad = jax.grad(rollout, argnums=(0, 1)) # NotImplementedError: Forward-mode differentiation rule for 'while' not implemented
    # gout = grad(x, us, A, B)
    # print(gout)


test_fori()


def test_scan():
    def rollout(x, us, A, B):
        def step(carry, u):
            # system dyamics
            next_x = A @ carry + B @ u
            carry = next_x
            return carry, next_x

        _, outs = scan(step, x, us)
        return np.square(outs).sum()

    A = np.array([[7.0, 0.5], [0.5, 0.8]])
    B = np.array([[0.4, 1.2]])
    x = np.ones(2)
    us = np.ones((10, 2))
    out = rollout(x, us, A, B)

    grad = jax.grad(rollout, argnums=(0, 1))
    gout = grad(x, us, A, B)
    print(gout)


test_scan()
