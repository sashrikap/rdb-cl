from jax import random
from rdb.infer import *


def test_1_chain():
    key = random.PRNGKey(1)
    proposal = IndGaussianProposal(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], proposal_var=5
    )
    proposal.update_key(key)
    data = {"a": 1, "b": 3}
    print(proposal(data))


def test_2_chainz():
    key = random.PRNGKey(1)
    proposal = IndGaussianProposal(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], proposal_var=5
    )
    proposal.update_key(key)
    data = [{"a": 1, "b": 3}, {"a": 1, "b": 5}]
    print(proposal(data))


def test_3_chainz():
    key = random.PRNGKey(1)
    proposal = IndGaussianProposal(
        rng_key=None, normalized_key="a", feature_keys=["a", "b"], proposal_var=5
    )
    proposal.update_key(key)
    data = [{"a": 1, "b": 3}] * 3
    print(proposal(data))
