"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] Pyro Implementation
    [2] Custon Implementation
Todo:
"""

import numpyro
import jax
from numpyro.util import control_flow_prims_disabled, fori_loop, optional
from numpyro.handlers import scale, condition, seed
from rdb.infer.algos import *
from rdb.optim.utils import concate_dict_by_keys
from rdb.infer.utils import logsumexp
from tqdm.auto import tqdm, trange
from scipy.stats import gaussian_kde
from time import time


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Attributes:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel):
        # self._kernel = seed(kernel, rng_key)
        self._kernel = kernel
        self._rng_key = rng_key

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def __call__(self, *args, **kwargs):
        """Likelihood method.

        Examples:
            >>> log_prob = self.likelihood(obs, *args)

        """
        return self._kernel(*args, **kwargs)


class IRDOptimalControl(PGM):
    """Inverse Reward Design for Optimal Control.

    Given environment `env`, `planner`, user input w
    infer p(w* | w).

    Notes:
        * Bayesian Terminology: theta -> true weights, obs -> proxy weights

    Methods:
        update : user select w
        infer : infer p(w* | obs)
        infogain : information gain (-entropy) in p(w* | obs)

    Example:
        >>> ird_oc = IRDOptimalControl(driving_env, runner, beta, prior_fn)
        >>> _, samples = ird_oc.posterior()

    """

    def __init__(self, rng_key, env, controller, runner, beta, prior_log_prob):
        """Construct IRD For optimal control.

        Args:
            controller: controller function, `actions = controller(state, weights)`
            runner: runner function, `traj, cost, info = runner(state, actions)`
            beta: temperature param: p ~ exp(beta * reward)

        """
        self._rng_key = rng_key
        self._controller = controller
        self._runner = runner
        self._env = env
        self._beta = beta
        self._prior_log_prob = seed(prior_log_prob, rng_key)
        self._normalizing_samples = {}
        self._best_actions = {}
        self._best_feats = {}
        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel)

    # def update(self, w_select):
    #    pass

    # def infer(self):
    #    pass

    # def infogain(self, env):
    #    pass

    def initialize(self, init_state, state_name, normalizer_weights):
        if state_name not in self._normalizing_samples.keys():
            self._normalizing_samples[state_name] = self._build_samples(
                init_state, normalizer_weights
            )

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._prior_log_prob = seed(self._prior_log_prob, rng_key)

    def _build_samples(self, init_state, normalizer_weights):
        sample_feats = []
        for normalizer_w in tqdm(
            normalizer_weights, desc="Collecting Normalizer Samples"
        ):
            actions = self._controller(init_state, weights=normalizer_w)
            xs, sample_cost, info = self._runner(
                init_state, actions, weights=normalizer_w
            )
            sample_feats.append(info["feats"])
        sample_feats = concate_dict_by_keys(sample_feats)
        return sample_feats

    def _build_kernel(self, beta):
        """Likelihood for Optimal Control.

        Args:
            prior_weights (dict): sampled from prior
            user_weights (dict): user-specified reward weights
            init_states (array): environment init state
            normalizing_samples(array): samples used to normalize likelihood in
                inverse reward design problem. Sampled before running `pgm.sample`.

        Example:
            >>> log_prob = likelihood_fn(weight, init_state)

        """

        @jax.jit
        def _likelihood(sample_weights, normalizing_samples, feats):
            """IRD Likelihood p(w_obs | w) with JAX speed up."""
            sample_cost = np.sum(
                [feats[key] * sample_weights[key] for key in feats.keys()]
            )
            sample_rew = -1 * sample_cost

            normalizing_rews = []
            for key in normalizing_samples.keys():
                # sum over episode
                w = sample_weights[key]
                rew = -1 * np.sum(w * normalizing_samples[key], axis=1)
                normalizing_rews.append(rew)

            ## Normalizing constant
            num_samples = len(normalizing_samples)
            # (n_samples, )
            log_norm_rew = logsumexp(np.sum(normalizing_rews, axis=0))
            log_norm_rew -= np.log(num_samples)

            log_prob = beta * (sample_rew - log_norm_rew)
            return log_prob

        def likelihood_fn(user_weights, sample_weights, state_name, init_state):
            assert (
                state_name in self._normalizing_samples.keys()
            ), "IRD model not initialized"
            normalizing_samples = self._normalizing_samples[state_name]
            if state_name in self._best_actions.keys():
                actions = self._best_actions[state_name]
                feats = self._best_feats[state_name]
            else:
                actions = self._controller(init_state, weights=user_weights)
                xs, sample_cost, info = self._runner(
                    init_state, actions, weights=sample_weights
                )
                feats = info["feats"]
                self._best_actions[state_name] = actions
                self._best_feats[state_name] = feats
            log_prob = _likelihood(sample_weights, normalizing_samples, feats)
            log_prob += self._prior_log_prob(sample_weights)
            return log_prob

        return likelihood_fn

    def entropy(self, data, method="gaussian", num_bins=100):
        if method == "gaussian":
            # scipy gaussian kde requires transpose
            kernel = gaussian_kde(data.T)
            N = data.shape[0]
            entropy = -(1.0 / N) * np.sum(np.log(kernel(data.T)))
        elif method == "histogram":
            raise NotImplementedError
        return entropy
