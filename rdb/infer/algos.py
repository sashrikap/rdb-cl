from jax import random
from numpyro.infer import MCMC, NUTS
import numpyro

"""Approximate Inference Algorithms.

Given p(y | theta), estimate p(theta | y)

Includes:
    * Metropolis-Hasting
    * Rejection Sampling
    * Hamiltonian Monte Carlo (numpyro-backed)
    * NUTS - No-U-Turn Monte Carlo (numpyro-backed)

Used for:
    * Inverse Reward Design

References:
    * Intro to Bayesisan stats: https://www.youtube.com/watch?v=OTO1DygELpY

Credits:
    * Jerry Z. He 2019
"""


class Inference(object):
    """ Generic Inference Class

    Notes:
        prior: p(theta), `kernel.prior`
        likelihood: p(obs | theta), `kernel.likelihood`

    """

    def __init__(self, kernel, num_samples, num_warmups):
        """ Construct Inference Object

        Args:
            kernel (fn): log_prob ~ Kernel(theta)

        """
        self._kernel = kernel
        self._num_samples = num_samples
        self._num_warmups = num_warmups
        self._key = random.PRNGKey(1)

    @property
    def num_samples(self):
        return self._num_samples

    def posterior(self, obs, *args, **kwargs):
        """Estimate p(theta | obs).

        Notes:
            [1] Estimate marginal p(obs)
            [2] Sample p(obs, theta) = p(obs | theta) p(theta)
            [3] Divide p(obs, theta) / p(obs)

        """
        raise NotImplementedError

    def marginal(self):
        """ Estimate p(obs) = sum_{theta} [p(obs | theta) p(theta)].

        Notes:
            [1] Randomly sample theta, condition on `obs`
            [2] Sum up samples

        """
        raise NotImplementedError


class NUTSMonteCarlo(Inference):
    def __init__(self, kernel, num_samples, num_warmups, step_size=1.0):
        super().__init__(kernel, num_samples, num_warmups)
        self._step_size = step_size

    def posterior(self, obs, *args, **kwargs):
        mcmc = MCMC(
            NUTS(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
        )
        mcmc.run(self._key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        posterior = None
        return posterior, samples


class HMCMonteCarlo(Inference):
    def __init__(self, kernel, num_samples, num_warmups, step_size=1.0):
        super().__init__(kernel, num_samples, num_warmups)
        self._step_size = step_size

    def posterior(self, obs, *args, **kwargs):
        mcmc = MCMC(
            HMC(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
        )
        mcmc.run(self._key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        posterior = None
        return posterior, samples


class MetropolisHasting(Inference):
    def __init__(self, kernel, num_samples):
        super().__init__(kernel, num_samples, num_warmups)

    def posterior(self, *args, **kwargs):
        posterior, samples = self.marginal()
        return posterior, samples

    def marginal(self):
        samples = []
        marginal = None
        return marginal, samples


class RejectionSampling(Inference):
    def __init__(self, kernel, num_samples, num_warmups):
        super().__init__(kernel, num_samples, num_warmups)

    def posterior(self, obs, *args, **kargs):
        pass
