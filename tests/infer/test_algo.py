import numpyro
import numpyro.distributions as dist
import numpy as onp
import copy
import jax.numpy as np
from rdb.infer import *
from rdb.exps.utils import Profiler
from jax import random, vmap
from numpyro.handlers import scale, condition, seed


def run_nuts():
    def kernel(obs):
        s = numpyro.sample("s", dist.Uniform(0.0, 10.0))
        z_fn = dist.Normal(s, 0.1)
        numpyro.sample(f"obs", z_fn, obs=obs)


class TestPrior(object):
    def __init__(self, rng_key=None):
        self._rng_key = rng_key

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._sample_fn = self._build_function()

    def log_prob(self, state):
        """ Uniformly 0 to 10, vectorized."""
        return np.where(
            np.logical_or(state < 0, state > 10.0),
            -np.ones_like(state) * np.inf,
            np.zeros_like(state),
        )

    def _build_function(self):
        def prior_fn():
            return numpyro.sample("prior", dist.Uniform(-10, 20))

        return seed(prior_fn, self._rng_key)

    def __call__(self):
        return self._sample_fn()


class TestProposal(object):
    def __init__(self, rng_key=None):
        self._rng_key = rng_key

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._proposal_fn = self._build_function()

    def _build_function(self):
        std = 0.05

        def proposal_fn(state):
            std = np.ones_like(state)
            return numpyro.sample("next_state", dist.Normal(state, std))

        return seed(proposal_fn, self._rng_key)

    def __call__(self, state):
        return self._proposal_fn(state)


def atest_mh_single():

    prior = TestPrior()

    def kernel(obs, state):
        """ Likelihood p(obs | s) """
        z_fn = dist.Normal(state, 0.1)
        log_prob = z_fn.log_prob(obs) + prior.log_prob(state)
        beta = 4.0
        return beta * log_prob

    key = random.PRNGKey(1)
    sampler = MetropolisHasting(
        None, kernel, num_warmups=40, num_samples=20, proposal=TestProposal()
    )
    sampler.update_key(key)
    samples = sampler.sample(obs=1.0, init_state=5.0, verbose=True)
    print(f"mean {np.mean(np.array(samples))}")


def test_mh_2_chainz():

    prior = TestPrior()

    def kernel(obs, state):
        """ Likelihood p(obs | s) """
        z_fn = dist.Normal(state, 0.1)
        log_prob = z_fn.log_prob(obs) + prior.log_prob(state)
        beta = 4.0
        return beta * log_prob

    key = random.PRNGKey(1)
    sampler = MetropolisHasting(
        None,
        kernel,
        num_warmups=40,
        num_samples=20,
        proposal=TestProposal(),
        num_chains=2,
    )
    sampler.update_key(key)
    samples = sampler.sample(obs=1.0, init_state=5.0, verbose=True)
    print(f"mean {np.mean(np.array(samples))}")


def test_mh_3_chainz():

    prior = TestPrior()

    def kernel(obs, state):
        """ Likelihood p(obs | s)

        Args:
            state (ndarray): vectoized state

        """
        z_fn = dist.Normal(state, 0.1)
        log_prob = z_fn.log_prob(obs) + prior.log_prob(state)
        beta = 4.0
        return beta * log_prob

    key = random.PRNGKey(1)
    sampler = MetropolisHasting(
        None,
        kernel,
        num_warmups=10,
        num_samples=20,
        proposal=TestProposal(),
        num_chains=3,
    )
    sampler.update_key(key)
    samples = sampler.sample(obs=1.0, init_state=5.0, verbose=True)
    print(f"mean {np.mean(np.array(samples))}")
