"""MCMC Inference Algorithms.

Given p(y | theta), estimate p(theta | y)

Includes:
    * Metropolis-Hasting
    * Rejection Sampling
    * Hamiltonian Monte Carlo (numpyro-backed)
    * NUTS - No-U-Turn Monte Carlo (numpyro-backed)

Used for:
    * Active Inverse Reward Design Experiments
    * Divide and Conquer IRD

Credits:
    * Jerry Z. He 2019-2020

"""

from jax import random
from numpyro.handlers import scale, condition, seed
from numpyro.infer import MCMC, NUTS
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
import jax
import copy
import numpyro
import numpy as onp
import jax.numpy as np
import numpyro.distributions as dist


class Inference(object):
    """ Generic Inference Class

    Args:
        kernel (fn)

    Example:
        >>> data = self.kernel() # p(obs | theta) p(theta)

    Notes:
        kernel: p(theta) p(obs | theta)

    """

    def __init__(self, rng_key, kernel, num_samples, num_warmups, num_chains=1):
        self._rng_key = rng_key
        self._kernel = kernel
        self._num_samples = num_samples
        self._num_warmups = num_warmups
        assert (
            isinstance(num_chains, int) and num_chains > 0
        ), f"Cannot use {num_chains} chains"
        self._num_chains = num_chains
        self._is_multi_chains = num_chains > 1

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def kernel(self):
        return self._kernel

    def update_key(self, rng_key):
        self._rng_key = rng_key

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
        * When sampling, initialize from observation.

    Args:
        num_samples (int): number of samples to return
        num_warmups (int): number of warmup iterations
        proposal (Proposal): given one sample, return next
        num_chains (int): if >=1, use 1st chain for sampling, others for convergence checking

    """

    def __init__(
        self, rng_key, kernel, num_samples, num_warmups, proposal, num_chains=1
    ):
        super().__init__(
            rng_key, kernel, num_samples, num_warmups, num_chains=num_chains
        )
        self._proposal = proposal
        self._coin_flip = None

    def _mh_step(self, obs, state, log_prob, verbose=False, *args, **kwargs):
        assert self._rng_key is not None, "Need to initialize with random key"

        next_state = self._proposal(state)
        next_log_prob = self._kernel(obs, next_state, **kwargs)
        log_ratio = next_log_prob - log_prob
        if verbose:
            # if True:
            print(
                "log next",
                next_log_prob,
                "log current",
                log_prob,
                "prob",
                onp.exp(log_ratio),
            )
        if self._is_multi_chains:
            accept = onp.log(self._coin_flip()) < log_ratio
            next_state = onp.where(accept, next_state, state)
            next_log_prob = onp.where(accept, next_log_prob, log_prob)
        else:
            # onp.where messes up with dictionary in single chain
            accept = onp.log(self._coin_flip()) < log_ratio
            if not accept:
                next_state = state
                next_log_prob = log_prob

        return accept, next_state, next_log_prob

    def _create_coin_flip(self):
        @jax.jit
        def raw_fn():
            return numpyro.sample("accept", dist.Uniform(0, 1))

        @jax.jit
        def raw_fn_multi():
            return numpyro.sample(
                "accept", dist.Uniform(0, 1), sample_shape=(self._num_chains,)
            )

        if self._is_multi_chains:
            return seed(raw_fn_multi, self._rng_key)
        else:
            return seed(raw_fn, self._rng_key)

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._proposal.update_key(rng_key)
        self._coin_flip = self._create_coin_flip()

    def _vectorize_state(self, state):
        if self._is_multi_chains:
            return np.array([state] * self._num_chains)
        else:
            return state

    def sample(
        self,
        obs,
        verbose=True,
        init_state=None,
        num_warmups=None,
        num_samples=None,
        name="",
        chain_viz=None,
        *args,
        **kwargs,
    ):
        """Vectorized MH Sample.

        Args:
            chain_viz (fn): visualize multiple chains, if provided. Called via `chain_viz(samples, accepts)`.

        """

        if init_state is not None:
            state = self._vectorize_state(init_state)
        else:
            state = self._vectorize_state(obs)
        obs = self._vectorize_state(obs)
        if num_warmups is None:
            num_warmups = self._num_warmups
        if num_samples is None:
            num_samples = self._num_samples

        log_prob = self._kernel(obs, state, **kwargs)

        # Warm-up phase
        warmup_accepts = []
        range_ = trange(self._num_warmups, desc="MH Warmup")
        for i in range_:
            try:
                accept, state, log_prob = self._mh_step(
                    obs, state, log_prob, verbose=False, *args, **kwargs
                )
            except:
                import pdb

                pdb.set_trace()
            warmup_accepts.append(accept)
            rate, num = self._get_counts(warmup_accepts, row=0)
            if verbose:
                range_.set_description(f"MH Warmup {name}; Accept {100 * rate:.1f}%")

        # Actual sampling phase (idential to warmup)
        samples = []
        accepts = []
        pbar = tqdm(total=num_samples, desc="MH Sampling")
        num = 0
        while not num == num_samples:
            accept, state, log_prob = self._mh_step(
                obs, state, log_prob, verbose=False, *args, **kwargs
            )
            accepts.append(accept)
            samples.append(state)
            rate, num = self._get_counts(accepts, row=0)
            pbar.n = num
            pbar.last_print_n = num
            pbar.refresh()
            pbar.set_description(f"MH Sampling {name}; Accept {100 * rate:.1f}%")
        self._summarize(accepts, samples, name)
        return self._select_samples(accepts, samples, chain_viz)

    def _get_counts(self, accepts, row=0):
        """Calculate acceptance rate.

        Args:
            accepts (list): one or multiple chains.

        """
        accepts = onp.array(accepts)
        if self._is_multi_chains:
            # Take first row
            assert row < accepts.shape[0]
            rate = accepts[:, row].sum() / len(accepts[:, row])
            num = accepts[:, row].sum()
            return rate, num
        else:
            rate = accepts.sum() / len(accepts)
            num = accepts.sum()
        return rate, num

    def _summarize(self, accepts, samples, name):
        """Summary after one MCMC sample round.

        Args:
            accepts (list): one or multiple chains.

        """
        accepts = onp.array(accepts)
        if self._is_multi_chains:
            for ir in range(accepts.shape[1]):
                rate, num = self._get_counts(accepts, row=ir)
                print(f"{name} MH chain {ir} rate {rate:.3f} accept {num}")
        else:
            rate, num = self._get_counts(accepts)
            print(f"{name} MH chain (single) rate {rate:.3f} accept {num}")

    def _select_samples(self, accepts, samples, chain_viz=None):
        accepts = onp.array(accepts)
        samples = onp.array(samples)
        if self._is_multi_chains:
            # Viz all rows
            if chain_viz is not None:
                chain_viz(samples, accepts)
            # Take first row
            samples = samples[accepts[:, 0]]
            assert samples.shape[0] == self._num_samples
            return samples
        else:
            samples = samples[accepts]
            assert len(samples) == self._num_samples
            return samples


class NUTSMonteCarlo(Inference):
    """No-U-Turn Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires kernel to provide gradient information

    """

    def __init__(
        self, rng_key, kernel, num_samples, num_warmups, step_size=1.0, num_chains=1
    ):
        super().__init__(rng_key, kernel, num_samples, num_warmups, num_chains=1)
        self._step_size = step_size

    def sample(self, obs, *args, **kwargs):
        raise NotImplementedError("Not completed")
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
        * Requires kernel to provide gradient information

    """

    def __init__(self, rng_key, kernel, num_samples, num_warmups, step_size=1.0):
        super().__init__(rng_key, kernel, num_samples, num_warmups)
        self._step_size = step_size

    def sample(self, obs, *args, **kwargs):
        raise NotImplementedError("Not completed")
        mcmc = MCMC(
            HMC(self._kernel, step_size=self._step_size),
            num_warmup=self._num_warmups,
            num_samples=self._num_samples,
        )
        mcmc.run(self._rng_key, obs, *args, **kwargs)
        samples = mcmc.get_samples()
        return samples


class RejectionSampling(Inference):
    def __init__(self, rng_key, kernel, num_samples, num_warmups):
        super().__init__(rng_key, kernel, num_samples, num_warmups)

    def sample(self, obs, *args, **kargs):
        raise NotImplementedError("Not completed")
