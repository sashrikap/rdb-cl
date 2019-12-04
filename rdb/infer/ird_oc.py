"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] Pyro Implementation
    [2] Custon Implementation
Todo:
"""

import numpyro
import jax
from numpyro.util import control_flow_prims_disabled, fori_loop, optional
from rdb.infer.algos import *


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Attributes:
        prior (fn): p(theta)
        likelihood (fn): p(obs | theta)
        sampler (object): MCMC Sampler specified by `method` (NUTS, etc.)

    """

    def __init__(self, prior_fn, likelihood_fn, method=None, sampler_args={}):
        """Construct PGM object.

        Args:
            method (str): sampling method (NUTS, HMC)

        Note:
            * Key functions implemented functional programming style: `prior_fn`, `likelihood_fn`

        """
        self._prior_fn = prior_fn
        self._likelihood_fn = likelihood_fn
        self._kernel = self._create_kernel()
        self._method = method
        self._sampler = self._create_sampler(sampler_args)

    @property
    def prior_fn(self):
        """Prior method.

        Examples:
            >>> prior = self.prior_fn()

        """
        return self._prior_fn

    @property
    def likelihood_fn(self):
        """Likelihood method.

        Args:
            prior (ndarray) : output from `prior_fn`
            obs (ndarray) : observation

        Examples:
            >>> prior = self.prior_fn()
            >>> log_prob = self.likelihood_fn(prior, obs, *args)

        """
        return self._likelihood_fn

    @property
    def kernel(self):
        return self._kernel

    def posterior(self, *args, **kargs):
        return self._sampler.posterior(*args, **kargs)

    def _create_kernel(self):
        def kernel_fn(data, *args, **kargs):
            prior = self.prior_fn()
            log_prob = self.likelihood_fn(prior, data, *args, **kargs)
            numpyro.factor("log_prob", log_prob)

        return kernel_fn

    def _create_sampler(self, sampler_args):
        if self._method == "MH":
            return MHMonteCarlo(self.kernel, **sampler_args)
        elif self._method == "NUTS":
            return NUTSMonteCarlo(self.kernel, **sampler_args)
        elif self._method == "HMC":
            return HMCMonteCarlo(self.kernel, **sampler_args)
        elif self._method is None:
            return None
        else:
            raise NotImplementedError(f"Unknown method {self._method}")


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

    def __init__(
        self,
        env,
        controller,
        runner,
        beta,
        prior_fn,
        method="NUTS",
        sampler_args={"num_samples": 100, "num_warmups": 100},
    ):
        """Construct IRD For optimal control.

        Args:
            controller: controller function, `actions = controller(state, weights)`
            runner: runner function, `traj, cost, info = runner(state, actions)`
            beta: temperature param: p ~ exp(beta * reward)

        """
        self._all_obs = []
        self._controller = controller
        self._runner = runner
        self._env = env
        self._beta = beta
        likelihood_fn = self._build_likelihood(beta)
        # Avoid numpyro's default jit arguments
        sampler_args["jit_model_args"] = False

        super().__init__(prior_fn, likelihood_fn, method, sampler_args)

    def update(self, w_select):
        pass

    def infer(self):
        pass

    def infogain(self, env):
        pass

    def _build_likelihood(self, beta):
        """Likelihood for Optimal Control.

        Args:
            prior_weights (dict): sampled from prior
            user_weights (dict): user-specified reward weights
            init_states (array): environment init state

        Example:
            >>> log_prob = likelihood_fn(weight, init_state)

        """

        def likelihood_fn(prior_weights, user_weights, init_state):
            print(f"prior weights {prior_weights}")
            actions = self._controller(init_state, weights=prior_weights)
            xs, user_cost, info = self._runner(
                init_state, actions, weights=prior_weights
            )
            feats_sum = info["feats_sum"]
            prior_cost, prior_costs = self._runner.compute_cost(
                xs, actions, weights=prior_weights
            )
            prior_rew = -1 * prior_cost
            return beta * prior_rew

        return likelihood_fn

    def _compute_infogain(self):
        pass
