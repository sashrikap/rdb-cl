import jax
import numpy as np
import jax.numpy as np
from jax import random
from rdb.infer import DictList
from jax.scipy.special import logsumexp

rng_key = random.PRNGKey(2)


@jax.jit
def sample_idx():
    arr = random.uniform(rng_key, (40, 40))
    sum_arr = np.sum(arr, axis=0)
    min_idx = np.argmin(arr)
    return arr[min_idx]


out = sample_idx()
assert out.shape == (40,)


def soft_max(costs, axis=None):
    costs = np.array(costs, dtype=float)
    return np.sum(np.exp(costs - logsumexp(costs, axis=axis)) * costs, axis=axis)


print(soft_max([1, 2, 3]))
print(soft_max([1, 2, 30]))
print(soft_max([10, 2, 3]))
