from jax import random
from rdb.infer import *
import numpy as onp
import pytest


@pytest.mark.parametrize("num", [1, 2, 3, 4])
def test_chain(num):
    key = random.PRNGKey(1)
    proposal = IndGaussianProposal(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], proposal_var=5
    )
    proposal.update_key(key)
    data = {"a": onp.ones(num), "b": 3 * onp.ones(num)}
    out = proposal(data)
    for key, val in out.items():
        assert len(val) == num
