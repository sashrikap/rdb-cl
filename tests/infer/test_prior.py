from jax import random
from rdb.infer import *
import numpy as onp
import pytest


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_chain(num):
    key = random.PRNGKey(1)
    prior = LogUniformPrior(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
    )
    prior.update_key(key)
    data = DictList({"a": onp.ones(num), "b": 3 * onp.ones(num)})
    prob = prior.log_prob(data)
    assert len(prob) == num


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_sample(num):
    key = random.PRNGKey(1)
    prior = LogUniformPrior(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
    )
    prior.update_key(key)
    data = prior(num)
    for key, val in data.items():
        assert len(val) == num
