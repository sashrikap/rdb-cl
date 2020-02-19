import numpy as onp
import jax.numpy as np
import jax.lax as lax
import jax
from time import time


def test_while_random():
    def body(x):
        i, val, rng_key = x
        rng_key, rng_random = jax.random.split(rng_key, 2)
        val += jax.random.uniform(rng_random)
        return (i + 1, val, rng_key)

    def cond(x):
        i, val, rng_key = x
        return i < 10

    i, val, rng_key = lax.while_loop(cond, body, (0, 0.0, jax.random.PRNGKey(0)))


def test_while_true():
    u = [1]

    def body(val):
        u[0] += np.sum(u)
        return val + 1

    cond_fn = lambda val: val < 1e9

    @jax.jit
    def loop():
        lax.while_loop(cond_fn, body, 0)

    t1 = time()
    loop()
    print(f"Time {time() - t1:.3f}")
    t1 = time()
    loop()
    print(f"Time {time() - t1:.3f}")


def test_fori_loop():
    @jax.jit
    def _batch_by_normal(data_a, data_b):
        """Multiply two very costly array in likelihood_fn.
        """
        # return data_a * data_b
        def _mult_add(i, sum_):
            return sum_ + data_a[i] * data_b[i]

        nfeats = len(data_a)
        sum_ = np.zeros(data_a.shape[1:])
        sum_ = jax.lax.fori_loop(0, nfeats, _mult_add, sum_)
        return sum_

    data_a = onp.random.random((5, 4, 3, 2))
    data_b = onp.random.random((5, 4, 3, 2))
    sum_ = _batch_by_normal(np.array(data_a), np.array(data_b))
    result = onp.sum(data_a * data_b, axis=0)
    assert onp.allclose(sum_, result)
