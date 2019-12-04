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


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Attributes:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel):
        # self._kernel = seed(kernel, rng_key)
        self._kernel = kernel
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
        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel)

    # def update(self, w_select):
    #    pass

    # def infer(self):
    #    pass

    # def infogain(self, env):
    #    pass

    def _build_kernel(self, beta):
        """Likelihood for Optimal Control.

        Args:
            prior_weights (dict): sampled from prior
            user_weights (dict): user-specified reward weights
            init_states (array): environment init state

        Example:
            >>> log_prob = likelihood_fn(weight, init_state)

        """

        def likelihood_fn(user_weights, sample_weights, init_state):
            actions = self._controller(init_state, weights=user_weights)
            xs, sample_cost, info = self._runner(
                init_state, actions, weights=sample_weights
            )
            # feats_sum = info["feats_sum"]
            # prior_cost, prior_costs = self._runner.compute_cost(
            #    xs, actions, weights=sample_weights
            # )
            log_prob = -1 * sample_cost
            log_prob += self._prior_log_prob(sample_weights)
            return beta * log_prob

        return likelihood_fn
