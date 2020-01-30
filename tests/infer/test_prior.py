from jax import random
from rdb.infer import *


def test_1_chain():
    key = random.PRNGKey(1)
    prior = LogUniformPrior(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
    )
    prior.update_key(key)
    data = {"a": 1, "b": 3}
    print(prior.log_prob(data))


def test_2_chainz():
    key = random.PRNGKey(1)
    prior = LogUniformPrior(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
    )
    prior.update_key(key)
    data = [{"a": 1, "b": 3}, {"a": 1, "b": 5}]
    print(prior.log_prob(data))


def test_3_chainz():
    key = random.PRNGKey(1)
    prior = LogUniformPrior(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
    )
    prior.update_key(key)
    data = [{"a": 1, "b": 3}] * 3
    print(prior.log_prob(data))
