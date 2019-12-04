import numpyro
import numpyro.distributions as dist
import numpy as onp
import copy
import jax.numpy as np
from jax import random, vmap
from rdb.infer.algos import *
from rdb.infer.ird_oc import PGM


def test_nuts():
    def kernel(obs):
        s = numpyro.sample("s", dist.Uniform(0.0, 10.0))
        z_fn = dist.Normal(s, 0.1)
        numpyro.sample(f"obs", z_fn, obs=obs)

    # infer = NUTSMonteCarlo(kernel, 100, 1000)
    # marginal, samples = infer.posterior(1.0)
    # import pdb; pdb.set_trace()


def test_mh():
    # def prior():
    #    s = numpyro.sample("s", dist.Uniform(0.0, 10.0))
    #    return s

    def prior_log_prob(state):
        if state < 0 or state > 10.0:
            return -np.inf
        else:
            return 0

    def kernel(obs, state):
        """ Likelihood p(obs | s) """
        z_fn = dist.Normal(state, 0.1)
        log_prob = z_fn.log_prob(obs) + prior_log_prob(state)
        beta = 4.0
        return beta * log_prob

    def proposal(state):
        std = 0.05
        next_state = numpyro.sample("next_state", dist.Normal(state, std))
        return next_state

    key = random.PRNGKey(1)
    pgm = PGM(key, kernel)
    sampler = MetropolisHasting(
        key, pgm, num_warmups=100, num_samples=200, proposal_fn=proposal
    )
    sampler.init(5.0)
    samples = sampler.sample(1.0)
    print(f"mean {np.mean(samples)}")
