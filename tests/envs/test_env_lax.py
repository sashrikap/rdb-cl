import jax.numpy as np
import jax.lax as lax
import jax
from time import time


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
