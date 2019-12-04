from jax import random, vmap
import jax.numpy as np

import pyro
import pyro.distributions as pdist
import torch
import numpyro
import numpyro.distributions as ndist
from numpyro.handlers import scale, condition, seed, trace
from numpyro.infer import MCMC, NUTS


def test_condition():
    def model(rng_key=None, sample_shape=(1,)):
        s = numpyro.sample("s", ndist.Uniform(0.0, 10.0), rng_key=rng_key)
        z = numpyro.sample(
            "z", ndist.Normal(s, 0.1), rng_key=rng_key, sample_shape=sample_shape
        )
        return z

    numpyro.sample("sample2", model2, obs, key, shape)
    tr = trace(condition(seed(model2, key), {"z": 1.0})).get_trace()
    tr["s"]["value"]


def test_factor_nuts():
    def kernel(rng_key=None, sample_shape=(1,)):
        s = numpyro.sample("s", ndist.Uniform(0.0, 10.0), rng_key=rng_key)
        z = numpyro.sample(
            "z", ndist.Normal(s, 0.1), rng_key=rng_key, sample_shape=sample_shape
        )
        obs = 1.0
        beta = 5
        diff = np.abs(z - obs)
        numpyro.factor("obs_log_prob", -diff * beta)

    key = random.PRNGKey(1)
    obs = 1.0
    shape = (100,)
    import pdb

    pdb.set_trace()
    kernel = NUTS(kernel)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=1000)
    mcmc.run(key)


def test_condition_nuts():
    def kernel(rng_key=None, sample_shape=(1,)):
        s = numpyro.sample("s", ndist.Uniform(0.0, 10.0), rng_key=rng_key)
        z = numpyro.sample(
            "z", ndist.Normal(s, 0.1), rng_key=rng_key, sample_shape=sample_shape
        )
        obs = 1.0
        beta = 5
        diff = np.abs(z - obs)
        numpyro.factor("obs_log_prob", -1.0 - diff * beta)

    key = random.PRNGKey(1)
    obs = 1.0
    shape = (100,)
    kernel = NUTS(kernel)
    mcmc = MCMC(kernel, num_warmup=100, num_samples=1000)
    mcmc.run(key)


def test_factor_hmc():
    return
