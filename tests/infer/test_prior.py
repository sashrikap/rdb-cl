from jax import random
from rdb.infer import *
import numpy as onp
import pytest


# @pytest.mark.parametrize("num", [1, 2, 3, 4])
# def test_chain(num):
#     key = random.PRNGKey(1)
#     prior = LogUniformPrior(
#         rng_key=None, normalized_key="a", feature_keys=["a", "b"], log_max=5
#     )
#     prior.update_key(key)
#     data = DictList({"a": onp.ones(num), "b": 3 * onp.ones(num)})


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_sample(num):
    prior = LogUniformPrior(normalized_key="a", feature_keys=["a", "b"], log_max=5)
    prior_0 = seed(prior, random.PRNGKey(0))
    data_0 = prior_0(num)

    prior_1 = seed(prior, random.PRNGKey(1))
    data_1 = prior_1(num)

    prior_0 = seed(prior, random.PRNGKey(0))
    data_0 = prior_0(num)

    for key, val in data_0.items():
        assert len(val) == num
