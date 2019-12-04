from jax import random
from numpyro.handlers import scale, condition, seed
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm, trange
import numpyro
import numpyro.distributions as dist
import jax.numpy as np

"""MCMC Inference Algorithms.

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
        model (fn)

    Example:
        >>> data = self.model() # p(obs | theta) p(theta)

    Notes:
        model: p(theta) p(obs | theta)

    """

    def __init__(self, rng_key, model, num_samples, num_warmups):
        self._rng_key = rng_key
        # self._model = seed(model, rng_seed=rng_key)
        self._model = model
        self._num_samples = num_samples
        self._num_warmups = num_warmups
        self._init_state = None

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def model(self):
        return self._model

    def init(self, state):
        self._init_state = state

    def sample(self, obs, *args, **kwargs):
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


class MetropolisHasting(Inference):
    """Metropolis-Hasting Algorithm.

    Note:
        * Basic Implementation.

    """

    def __init__(
        self, rng_key, model, num_samples, num_warmups, proposal_fn, step_size=1.0
    ):
        super().__init__(rng_key, model, num_samples, num_warmups)
        self._step_size = step_size
        self._proposal = seed(proposal_fn, rng_seed=rng_key)
        self._coin_flip = self._create_coin_flip()

    def _mh_step(self, obs, state, log_prob, *args, **kwargs):
        next_state = self._proposal(state)
        next_log_prob = self._model(obs, next_state, **kwargs)
        log_ratio = next_log_prob - log_prob
        accept = self._coin_flip() < np.exp(log_ratio)
        if not accept:
            next_state = state
            next_log_prob = log_prob
        return accept, next_state, next_log_prob

    def _create_coin_flip(self):
        def fn():
            return numpyro.sample("accept", dist.Uniform(0, 1))

        return seed(fn, self._rng_key)

    def init(self, state):
        self._init_state = state

    def sample(self, obs, verbose=True, *args, **kwargs):
        assert self._init_state is not None, "Need to initialize"
        state = self._init_state
        log_prob = self._model(obs, state, **kwargs)
        range_ = range(self._num_warmups)
        if verbose:
            range_ = trange(self._num_warmups, desc="MH Warmup")
        for i in range_:
            _, state, log_prob = self._mh_step(obs, state, log_prob, **kwargs)

        samples = []
        accepts = []
        ratio = 0.0
        num_steps = 0
        range_ = range(self._num_samples)
        if verbose:
            range_ = trange(self._num_samples, desc="MH Sampling")
        for i in range_:
            accept = False
            while not accept:
                num_steps += 1
                accept, state, log_prob = self._mh_step(obs, state, log_prob, **kwargs)
                accepts.append(accept)
            ratio = float(np.sum(accepts)) / len(accepts)
            if verbose:
                range_.set_description(f"MH Sampling; Accept {100 * ratio:.1f}%")
            samples.append(state)
        if verbose:
            print(f"Acceptance ratio {ratio}")
        self._init_state = None
        return samples


class NUTSMonteCarlo(Inference):
    """No-U-Turn Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires kernel to provide gradient information

    """

    def __init__(self, rng_key, model, num_samples, num_warmups, step_size=1.0):
        super().__init__(rng_key, model, num_samples, num_warmups)
        self._step_size = step_size

    def sample(self, obs, *args, **kwargs):
        mcmc = MCMC(
            NUTS(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
        )
        mcmc.run(self._rng_key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        return samples


class HamiltonionMonteCarlo(Inference):
    """Hamiltonion Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires model to provide gradient information

    """

    def __init__(self, rng_key, model, num_samples, num_warmups, step_size=1.0):
        super().__init__(rng_key, model, num_samples, num_warmups)
        self._step_size = step_size

    def sample(self, obs, *args, **kwargs):
        mcmc = MCMC(
            HMC(self._model, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
        )
        mcmc.run(self._rng_key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        return samples


class RejectionSampling(Inference):
    def __init__(self, rng_key, model, num_samples, num_warmups):
        super().__init__(rng_key, model, num_samples, num_warmups)

    def sample(self, obs, *args, **kargs):
        pass
