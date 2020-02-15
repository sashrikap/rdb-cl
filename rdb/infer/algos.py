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

from numpyro.handlers import scale, condition, seed
from rdb.infer.dictlist import DictList
from numpyro.infer import MCMC, NUTS
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from jax import random
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

    def __init__(
        self, rng_key, kernel, prior, proposal, num_samples, num_warmups, num_chains=1
    ):
        self._rng_key = rng_key
        self._kernel = kernel
        self._prior = prior
        self._proposal = proposal
        self._num_samples = num_samples
        self._num_warmups = num_warmups
        assert (
            isinstance(num_chains, int) and num_chains > 0
        ), f"Cannot use {num_chains} chains"
        self._num_chains = num_chains

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
        * By default, state uses np.array. Also supports rdb.infer.dictlist

    Args:
        num_samples (int): number of samples to return
        num_warmups (int): number of warmup iterations
        proposal (Proposal): given one sample, return next
        num_chains (int): if >=1, use 1st chain for sampling, others for convergence checking
        kernel (fn): inference kernel
        use_dictlist (bool): state represented by DictList

    Kernel args:
        obs (ndarray): (obs_dim, )
        state (ndarray): (obs_dim, )
        tasks (ndarray): (ntasks, )

    """

    def __init__(
        self,
        rng_key,
        kernel,
        prior,
        proposal,
        num_samples,
        num_warmups,
        num_chains=1,
        use_dictlist=False,
    ):
        super().__init__(
            rng_key,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            num_samples=num_samples,
            num_warmups=num_warmups,
            num_chains=num_chains,
        )
        self._coin_flip = None
        self._use_dictlist = use_dictlist

    def _concatenate(self, array_a, array_b):
        """DictList does not support np.concatenate."""
        if self._use_dictlist and isinstance(array_a, DictList):
            return array_a.concat(array_b)
        else:
            return np.concatenate([array_a, array_b])

    def _vectorize_state(self, state):
        """Vectorize for batch sampling.

        Args:
            state: (state_dim, ...)

        Output:
            state: (nchains, state_dim, ...)

        """
        if self._use_dictlist:
            #  shape nfeats * (nstate, ...)
            return DictList([state for _ in range(self._num_chains)], jax=True)
        else:
            return np.repeat(state[None, :], self._num_chains, axis=0)

    def _mh_step(self, obs, state, logpr, *args, **kwargs):
        """Metropolis hasing step.

        Args:
            obs (ndarray): (xdim, ...)
            state (ndarray): current sample state (nbatch, xdim, ...)

        Output:
            accept (bool array): (nbatch,)
            next_state (ndarray): (nbatch, xdim, ...)
            next_log_prob: (nbatch,)

        """

        assert self._rng_key is not None, "Need to initialize with random key"
        assert len(state) == self._num_chains

        ## Sample next state (nbatch, xdim)
        # with Profiler("MH Proposal"):
        next_state = self._proposal(state)
        # with Profiler("MH Kernel 1"):
        next_logpr = self._kernel(obs, next_state, **kwargs)
        # with Profiler("MH Others 1.1"):
        # with Profiler("MH Others 1.4"):
        logp_ratio = next_logpr - logpr
        # print(type(next_logpr), type(logpr))
        # with Profiler("MH Others 1.5"):
        ## Accept or not (nbatch,)
        coin_flip = np.log(self._coin_flip())

        if False:
            # if True:
            print(f"next {next_logpr} curr {logpr} ratio {logp_ratio} flip {coin_flip}")
        # with Profiler("MH Others 2"):
        accept = coin_flip < np.array(logp_ratio)

        ## Check and mask out -inf
        pos_inf = np.logical_and(np.isinf(next_logpr), next_logpr > 0)
        neg_inf = np.isneginf(next_logpr)
        assert not any(pos_inf)
        # with Profiler("MH Others 3"):
        next_logpr = jax.ops.index_update(next_logpr, neg_inf, 0.0)
        # next_logpr[neg_inf] = 0.0

        ## Accept for next state
        # with Profiler("MH Others 4"):
        not_accept = np.logical_not(accept)
        #  shape (nbatch, )
        next_logpr = next_logpr * accept + logpr * not_accept
        #  shape (nbatch, xdim)
        next_state = next_state * accept + state * not_accept
        # with Profiler("MH Others 5"):
        assert next_logpr.shape == (self._num_chains,)
        assert next_state.shape == state.shape
        return accept, next_state, next_logpr

    def _create_coin_flip(self):
        def raw_fn():
            return numpyro.sample("accept", dist.Uniform(0, 1))

        def raw_fn_multi():
            return numpyro.sample(
                "accept", dist.Uniform(0, 1), sample_shape=(self._num_chains,)
            )

        return seed(raw_fn_multi, self._rng_key)

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._proposal.update_key(rng_key)
        self._coin_flip = self._create_coin_flip()

    def sample(
        self,
        obs,
        init_state=None,
        num_warmups=None,
        num_samples=None,
        name="",
        *args,
        **kwargs,
    ):
        """Vectorized MH Sample.

        Args:
            obs (ndarray/DictList): observation
                shape (ndarray): (state_dim,)
                shape (DictList): nkeys * (state_dim,)
            init_state (xdim,), use expand_dims=True

        """

        if num_warmups is None:
            num_warmups = self._num_warmups
        if num_samples is None:
            num_samples = self._num_samples
        ## Preprocessing
        state = init_state if init_state is not None else obs

        ## Expand dimension into batch-first
        obs = self._vectorize_state(obs)
        state = self._vectorize_state(state)
        log_prob = self._kernel(obs, state, **kwargs)
        assert log_prob.shape == (self._num_chains,)

        ## Warm-up phase
        warmup_accepts = []
        print(f"MH Samplint Chains={self._num_chains}")
        pbar = tqdm(total=self._num_warmups, desc="MH Warmup")
        for i in range(self._num_warmups):
            accept, state, log_prob = self._mh_step(
                obs, state, log_prob, *args, **kwargs
            )
            log_prob = np.array(log_prob)
            warmup_accepts.append(accept)
            rate, num = self._get_counts(warmup_accepts, chain=0)
            pbar.n, pbar.last_print_n = i + 1, i + 1
            pbar.refresh()
            pbar.set_description(f"MH Warmup {name}; Accept {100 * rate:.1f}%")

        ## Actual sampling phase (idential to warmup)
        samples = []
        accepts = []
        pbar = tqdm(total=num_samples, desc="MH Sampling")
        num = 0
        nsteps = 0
        while not num == num_samples:
            accept, state, log_prob = self._mh_step(
                obs, state, log_prob, *args, **kwargs
            )
            accepts.append(accept)
            samples.append(state)
            nsteps += 1
            rate, num = self._get_counts(accepts, chain=0)
            pbar.n, pbar.last_print_n = num, num
            pbar.refresh()
            pbar.set_description(f"MH Sampling {name}; Accept {100 * rate:.1f}%")

        ## Check multi-chain result shapes
        #  samples (nsteps, self._num_chains, xdim...)
        if self._use_dictlist:
            samples = DictList(samples)
        else:
            samples = np.array(samples)
        #  accepts (nsteps, self._num_chains)
        accepts = np.array(accepts)
        assert accepts.shape == (nsteps, self._num_chains)
        ## Summarize and select accepted
        rates = self._summarize(samples, accepts, name)
        accepted_chains = self._accepted_samples(samples, accepts)
        info = {"all_chains": accepted_chains, "rates": rates}
        main_chain = accepted_chains[0]
        return main_chain, info

    def _get_counts(self, accepts, chain=0):
        """Calculate acceptance rate.

        Args:
            accepts (list): one or multiple chains.

        """
        accepts = np.array(accepts)
        # Take first chain
        assert chain < accepts.shape[1]
        rate = accepts[:, chain].sum() / len(accepts[:, chain])
        num = accepts[:, chain].sum()
        return rate, num

    def _summarize(self, samples, accepts, name):
        """Summary after one MCMC sample round.

        Args:
            accepts (list): one or multiple chains.

        """
        accepts = np.array(accepts)
        rates = []
        for chain in range(self._num_chains):
            rate, num = self._get_counts(accepts, chain=chain)
            print(f"{name} MH chain {chain} rate {rate:.3f} accept {num}")
            rates.append(rate)
        return rates

    def _accepted_samples(self, samples, accepts):
        """Select accepted samples from the main chain.

        Args:
            samples (ndarray): (nsteps, nchains, xdim)
            accepts (ndarray): (nsteps, nchains, 1)

        """

        accepted_samples = []
        for chain in range(self._num_chains):
            chain_samples = samples[:, chain]
            chain_accepts = accepts[:, chain]
            accepted_samples.append(chain_samples[list(chain_accepts)])
        assert len(accepted_samples[0]) == self._num_samples
        return accepted_samples


class NUTSMonteCarlo(Inference):
    """No-U-Turn Monte Carlo Sampling.

    Note:
        * Backed by https://github.com/mfouesneau/NUTS
        * Requires kernel to provide gradient information

    """

    def __init__(
        self,
        rng_key,
        kernel,
        prior,
        proposal,
        num_samples,
        num_warmups,
        step_size=1.0,
        num_chains=1,
    ):
        super().__init__(
            rng_key,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            num_samples=num_samples,
            num_warmups=num_warmups,
            num_chains=num_chains,
        )
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
        return samples, {}


class HamiltonionMonteCarlo(Inference):
    """Hamiltonion Monte Carlo Sampling.

    Note:
        * Backed by numpyro implementation
        * Requires kernel to provide gradient information

    """

    def __init__(
        self, rng_key, kernel, prior, proposal, num_samples, num_warmups, step_size=1.0
    ):
        super().__init__(
            rng_key,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            num_samples=num_samples,
            num_warmups=num_warmups,
            num_chains=num_chains,
        )
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
        return samples, {}


class RejectionSampling(Inference):
    def __init__(self, rng_key, kernel, prior, proposal, num_samples, num_warmups):
        super().__init__(
            rng_key,
            kernel=kernel,
            prior=prior,
            proposal=proposal,
            num_samples=num_samples,
            num_warmups=num_warmups,
            num_chains=num_chains,
        )

    def sample(self, obs, *args, **kargs):
        raise NotImplementedError("Not completed")
