from jax import random
from numpyro.infer import MCMC, NUTS, MCMCKernel
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

    Args:
        prior (fn): `prior = f()`

    Notes:
        prior: p(theta)
        likelihood: p(obs | theta)

    """

    def __init__(self, prior, likelihood, num_samples, num_warmups, jit_args=True):
        self._prior = prior
        self._likelihood = likelihood
        self._num_samples = num_samples
        self._num_warmups = num_warmups
        self._key = random.PRNGKey(1)
        self._jit_args = jit_args
        self._kernel = self._create_kernel()

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

    def _create_kernel(self):
        """ Create sampling kernel function.

        Example:
            >>> data = kernel_fn() # p(obs | theta) p(theta)
        """
        raise NotImplementedError

    def marginal(self):
        """ Estimate p(obs) = sum_{theta} [p(obs | theta) p(theta)].

        Notes:
            [1] Randomly sample theta, condition on `obs`
            [2] Sum up samples

        """
        raise NotImplementedError


class MHMonteCarlo(Inference):
    """Metropolis-Hasting Algorithm.

    Note:
        * Basic Implementation.

    """

    def __init__(
        self,
        prior,
        likelihood,
        num_samples,
        num_warmups,
        jit_model_args=True,
        step_size=1.0,
    ):
        super().__init__(prior, likelihood, num_samples, num_warmups)
        self._step_size = step_size

    def _create_kernel(self):
        """Numpyro-based kernel."""

        def kernel_fn(data, *args, **kargs):
            prior = self._prior_fn()
            log_prob = self._likelihood_fn(prior, data, *args, **kargs)
            return log_prob

        return kernel_fn

    def _mh_step(self):
        pass

    def posterior(self, obs, *args, **kwargs):
        for i in range(self._num_warmups):
            pass

        samples = []
        for i in range(self._num_samples):
            pass
        return posterior, samples


class NUTSMonteCarlo(Inference):
    """No-U-Turn Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires kernel to provide gradient information

    """

    def __init__(
        self,
        prior,
        likelihood,
        num_samples,
        num_warmups,
        jit_model_args=True,
        step_size=1.0,
    ):
        super().__init__(prior, likelihood, num_samples, num_warmups)
        self._step_size = step_size

    def _create_kernel(self):
        """Numpyro-based kernel."""

        def kernel_fn(data, *args, **kargs):
            prior = self._prior_fn()
            log_prob = self._likelihood_fn(prior, data, *args, **kargs)
            numpyro.factor("log_prob", log_prob)

        return kernel_fn

    def posterior(self, obs, *args, **kwargs):
        mcmc = MCMC(
            NUTS(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
            jit_model_args=self._jit_args,
        )
        mcmc.run(self._key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        posterior = None
        return posterior, samples


class HMCMonteCarlo(Inference):
    """Hamiltonion Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires kernel to provide gradient information

    """

    def __init__(
        self, kernel, num_samples, num_warmups, jit_model_args=True, step_size=1.0
    ):
        super().__init__(kernel, num_samples, num_warmups)
        self._step_size = step_size

    def _create_kernel(self):
        """Numpyro-based kernel."""

        def kernel_fn(data, *args, **kargs):
            prior = self._prior_fn()
            log_prob = self._likelihood_fn(prior, data, *args, **kargs)
            numpyro.factor("log_prob", log_prob)

        return kernel_fn

    def posterior(self, obs, *args, **kwargs):
        mcmc = MCMC(
            HMC(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
            jit_model_args=self._jit_args,
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
