import jax.numpy as np
import numpy as onp
import tensorflow as tf
import timeit


def test_jax_tf():
    def slow_f(x):
        return x * x + x * 2.0

    fast_f_xla = tf.function(slow_f, experimental_compile=True)
    fast_f = tf.function(slow_f)

    x = np.ones((5000, 5000))
    N = int(1e3)
    print("Fast_f_xla first", timeit.timeit(lambda: fast_f_xla(x), number=1))
    print("Fast_f_xla", timeit.timeit(lambda: fast_f_xla(x), number=N) / N)
    print("Fast_f first", timeit.timeit(lambda: fast_f(x), number=1))
    print("Fast_f", timeit.timeit(lambda: fast_f(x), number=N) / N)
