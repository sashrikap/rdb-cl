"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] Pyro Implementation
    [2] Custon Implementation
Note:
    * We use `init_state/task` interchangably. In drive2d
     `env.set_task(init_state)` specifies a task to run.

"""

import numpyro
import jax
from numpyro.handlers import scale, condition, seed
from rdb.exps.utils import plot_samples
from rdb.infer.algos import *
from rdb.infer.particles import Particles
from rdb.infer.utils import logsumexp
from tqdm.auto import tqdm, trange
from time import time


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Methods:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel, proposal_fn, sample_method, sample_args):
        self._rng_key = rng_key
        self._sampler = self._build_sampler(
            kernel, proposal_fn, sample_method, sample_args
        )

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._sampler.update_key(rng_key)

    def _build_sampler(self, kernel, proposal_fn, sample_method, sample_args):
        if sample_method == "mh":
            return MetropolisHasting(
                self._rng_key, kernel, proposal_fn=proposal_fn, **sample_args
            )
        else:
            raise NotImplementedError


class IRDOptimalControl(PGM):
    """Inverse Reward Design for Optimal Control.

    Given environment `env`, `planner`, user input w
    infer p(w* | w).

    Notes:
        * Bayesian Terminology: theta -> true w, obs -> proxy w
        * Provides caching functions to avoid costly feature calcultions

    Args:
        controller (fn): controller function,
            `actions = controller(state, w)`
        runner (fn): runner function,
            `traj, cost, info = runner(state, actions)`
        beta (float): temperature param
            `p ~ exp(beta * reward)`
        prior_log_prob (fn): log probability of prior
        normalizer_fn (fn): sample fixed number of normalizers

    Methods:
        sample (fn): sample b(w) given obs_w
        get_samples (fn): get sample features on new task
        update (fn): incorporate user's obs_w
            next likelihood `p(w | obs_w1) p(w | obs_w2)...`
        infer (fn): infer p`(w* | obs)`

    Example:
        >>> ird_oc = IRDOptimalControl(driving_env, runner, beta, prior_fn)
        >>> _, samples = ird_oc.posterior()

    """

    def __init__(
        self,
        rng_key,
        env,
        controller,
        runner,
        beta,
        true_w,
        prior_log_prob,
        normalizer_fn,
        proposal_fn,
        sample_method="mh",
        sample_args={},  # "num_warmups": 100, "num_samples": 200
        designer_args={},
    ):
        self._rng_key = rng_key
        self._controller = controller
        self._runner = runner
        self._env = env
        self._beta = beta
        self._prior_log_prob = prior_log_prob
        self._normalizer_fn = normalizer_fn
        ## Caching normalizers, samples and features
        self._normalizer = None
        self._samples = {}
        self._user_actions = {}
        self._user_feats = {}
        # assume designer uses the same beta
        self._designer = Designer(
            rng_key,
            env,
            controller,
            runner,
            beta,
            true_w,
            proposal_fn,
            sample_method,
            designer_args,
        )
        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sample_args)

    def update_key(self, rng_key):
        """ Update random key """
        super().update_key(rng_key)
        self._designer.update_key(rng_key)
        self._prior_log_prob = seed(self._prior_log_prob, rng_key)
        self._normalizer_fn = seed(self._normalizer_fn, rng_key)

    def get_samples(self, task, task_name):
        """Sample features for belief weights on a task
        """
        assert task_name in self._samples.keys(), f"{task_name} not found"
        return self._samples[task_name]

    def update(self, obs):
        """ Incorporate new observation """
        pass

    def simulate_designer(self, task, task_name):
        """Sample one w from b(w) on task"""
        particles = self._designer.sample(task, task_name)
        return particles.subsample(1)

    def sample(self, task, task_name, obs, verbose=True, visualize=False):
        """Sample b(w) for true weights given obs.weights.

        Args:
            obs (Particle): 1 observation particle

        """
        if self._normalizer is None:
            self._normalizer = self._build_normalizer(task, task_name)
        user_feats_sum = obs.get_features_sum(task, task_name)
        norm_feats_sum = self._normalizer.get_features_sum(
            task, task_name, "Computing Normalizers"
        )
        print("Sampling IRD")
        sample_ws = self._sampler.sample(
            obs.weights[0],
            norm_feats_sum=norm_feats_sum,
            user_feats_sum=user_feats_sum,
            verbose=verbose,
        )
        samples = Particles(
            self._rng_key, self._env, self._controller, self._runner, sample_ws
        )
        self._samples[task_name] = samples
        if visualize:
            plot_samples(
                samples.weights,
                highlight_dict=obs.weights[0],
                save_path="data/samples.png",
            )

        return self._samples[task_name]

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    def _build_normalizer(self, task, task_name):
        """For new tasks, need to build normalizer."""
        norm_ws = self._normalizer_fn()
        state = self._get_init_state(task)
        normalizer = Particles(
            self._rng_key, self._env, self._controller, self._runner, norm_ws
        )
        norm_feats = normalizer.get_features(
            task, task_name, desc="Collecting Normalizer Samples"
        )
        return normalizer

    def _build_kernel(self, beta):
        """Likelihood for observed data.

        Args:
            prior_w (dict): sampled from prior
            user_w (dict): user-specified reward weight
            tasks (array): environment init state
            norm_feats_sum(array): samples used to normalize likelihood in
                inverse reward design problem. Sampled before running `pgm.sample`.

        Example:
            >>> log_prob = likelihood_fn(weight, task)

        """

        def likelihood_fn(user_w, sample_w, norm_feats_sum, user_feats_sum):
            """Main likelihood logic.

            Runs `p(w_obs | w)`

            """
            sample_rew = (
                -1
                * np.array(
                    [sample_w[key] * user_feats_sum[key] for key in sample_w.keys()]
                ).sum()
            )
            ## Normalizing constant
            normal_rews = -1 * np.array(
                [sample_w[key] * norm_feats_sum[key] for key in sample_w.keys()]
            ).sum(axis=0)
            N = len(normal_rews)
            sum_normal_rew = -np.log(N) + logsumexp(normal_rews)

            log_prob = beta * (sample_rew - sum_normal_rew)
            log_prob += self._prior_log_prob(sample_w)
            return log_prob

        return likelihood_fn


class Designer(PGM):
    """Simulated designer module. Given true weights, sample noisy rational design weights.

    Note:
        * Currently assumes same beta, proposal as IRD module, although
          doesn't have to be.
    """

    def __init__(
        self,
        rng_key,
        env,
        controller,
        runner,
        beta,
        true_w,
        proposal_fn,
        sample_method,
        sampler_args,
    ):
        self._rng_key = rng_key
        self._env = env
        self._controller = controller
        self._runner = runner
        self._true_w = true_w
        kernel = self._build_kernel(beta)
        self._samples = {}
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sampler_args)

    def sample(self, task, task_name, verbose=True):
        """Sample 1 set of weights from b(design_w) given true_w"""
        if task_name not in self._samples.keys():
            print("Sampling Designer")
            sample_ws = self._sampler.sample(self._true_w, task=task)
            samples = Particles(
                self._rng_key, self._env, self._controller, self._runner, sample_ws
            )
            self._samples[task_name] = samples
        return self._samples[task_name]

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    @property
    def true_w(self):
        return self._true_w

    def update_key(self, rng_key):
        super().update_key(rng_key)

    def _build_kernel(self, beta):
        def likelihood_fn(true_w, sample_w, task):
            state = self._get_init_state(task)
            actions = self._controller(state, weights=sample_w)
            _, cost, info = self._runner(state, actions, weights=true_w)
            rew = -1 * cost
            return beta * rew

        return likelihood_fn
