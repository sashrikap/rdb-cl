import numpyro
import numpyro.distributions as dist
from rdb.infer.algos import *


def test_nuts():
    def kernel(obs):
        s = numpyro.sample("s", dist.Uniform(0.0, 10.0))
        z = numpyro.sample("z", dist.Normal(s, 0.1))
        z_fn = dist.Normal(s, 0.1)
        numpyro.sample(f"obs", z_fn, obs=obs)

    infer = NUTSMonteCarlo(kernel, 100, 1000)
    marginal, samples = infer.posterior(1.0)
    # import pdb; pdb.set_trace()
