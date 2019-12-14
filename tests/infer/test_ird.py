from rdb.infer.ird_oc import *
from rdb.infer.algos import *
from rdb.infer.utils import *
import jax.numpy as np
import numpyro
from jax import random, vmap
import numpyro.distributions as dist


def ttest_PGM_kernel():
    prior_fn = lambda: numpyro.sample("s", dist.Uniform(0.0, 10.0))

    def likelihood(prior, obs, std):
        z = numpyro.sample("z", dist.Normal(prior, std))
        beta = 5.0
        diff = np.abs(z - obs)
        return -beta * diff

    pgm = PGM(prior_fn, likelihood)

    infer = NUTSMonteCarlo(pgm.kernel, 100, 100)
    _, samples = infer.posterior(obs=1.0, std=0.1)
    # import pdb; pdb.set_trace()
