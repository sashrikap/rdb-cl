"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] Pyro Implementation
    [2] Custon Implementation
Note:
    * We use `init_state/task` interchangably. In drive2d
     `env.set_task(init_state)` specifies a task to run.

"""

import numpyro
import copy
import jax
import jax.numpy as np
from numpyro.handlers import scale, condition, seed
from rdb.optim.utils import multiply_dict_by_keys
from rdb.infer.algos import *
from rdb.infer.particles import Particles
from rdb.exps.utils import plot_weights, Profiler
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
        # self._sampler.update_key(rng_key)

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
        controller_fn (fn): returns controller and runner
            controller: `actions = controller(state, w)`
            runner: `traj, cost, info = runner(state, actions)`
        beta (float): temperature param
            `p ~ exp(beta * reward)`
        prior_log_prob (fn): log probability of prior
        normalizer_fn (fn): sample fixed number of normalizers
        use_true_w (bool): debug designer with true w

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
        env_fn,
        controller_fn,
        eval_server,
        beta,
        true_w,
        prior_log_prob_fn,
        normalizer_fn,
        proposal_fn,
        sample_method="mh",
        sample_args={},  # "num_warmups": 100, "num_samples": 200
        designer_args={},
        use_true_w=False,
        debug_true_w=False,
        test_mode=True,  # Skip _cache_task part if true
    ):
        self._rng_key = rng_key
        self._env_fn = env_fn
        self._env = env_fn()
        self._controller, self._runner = controller_fn(self._env)
        self._eval_server = eval_server
        # Rationality
        self._beta = beta
        # Sampling functions
        self._prior_raw_fn = prior_log_prob_fn
        self._prior_log_prob = None
        self._normalizer_raw_fn = normalizer_fn
        self._normalizer_fn = None
        ## Caching normalizers, samples and features
        self._normalizer = None
        self._samples = {}
        self._user_actions = {}
        self._user_feats = {}
        # assume designer uses the same beta, prior and prior proposal
        self._designer = Designer(
            rng_key,
            self._env_fn,
            self._controller,
            self._runner,
            beta,
            true_w,
            prior_log_prob,
            proposal_fn,
            sample_method,
            designer_args,
            use_true_w=use_true_w,
        )
        self._debug_true_w = debug_true_w
        self._test_mode = test_mode

        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sample_args)

    @property
    def env(self):
        return self._env

    @property
    def env_fn(self):
        return self._env_fn

    @property
    def designer(self):
        return self._designer

    @property
    def eval_server(self):
        return self._eval_server

    def update_key(self, rng_key):
        """ Update random key """
        super().update_key(rng_key)
        self._sampler.update_key(rng_key)
        self._designer.update_key(rng_key)
        self._prior_log_prob = seed(self._prior_raw_fn, rng_key)
        self._normalizer_fn = seed(self._normalizer_raw_fn, rng_key)

    def load_samples(self,):
        pass

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
        sub_sample = particles.subsample(1)
        return sub_sample

    def sample(self, tasks, task_names, obs, verbose=True, visualize=False):
        """Sample b(w) for true weights given obs.weights.

        Args:
            tasks (list): list of tasks so far
            task_names (list): list of task names so far
            obs (list): list of observation particles, each for 1 task

        Note:
            * Samples are cached by the last task name to date (`task_names[-1]`)

        TODO:
            * currently `tasks`, `task_names`, `obs` are ugly lists

        """
        if self._normalizer is None:
            self._normalizer = self._build_normalizer(tasks[0], task_names[0])
        all_obs_ws = []
        all_user_feats_sum, all_norm_feats_sum = [], []
        for task_i, name_i, obs_i in zip(tasks, task_names, obs):
            all_obs_ws.append(obs_i.weights[0])
            all_user_feats_sum.append(obs_i.get_features_sum(task_i, name_i))
            all_norm_feats_sum.append(
                self._normalizer.get_features_sum(
                    task_i, name_i, "Computing Normalizers"
                )
            )
        last_obs_w = obs[-1].weights[0]
        last_name = task_names[-1]

        print("Sampling IRD")
        if self._test_mode:
            sample_ws = [
                copy.deepcopy(obs[0].weights[0])
                for _ in range(self._sampler.num_samples)
            ]
        else:
            sample_ws = self._sampler.sample(
                obs=all_obs_ws,  # all obs so far
                init_state=last_obs_w,  # initialize with last obs
                user_feats_sums=all_user_feats_sum,
                norm_feats_sums=all_norm_feats_sum,
                verbose=verbose,
            )
        samples = Particles(
            self._rng_key,
            self._env_fn,
            self._controller,
            self._runner,
            sample_ws,
            test_mode=self._test_mode,
        )
        self._samples[last_name] = samples
        # if visualize:
        #     plot_weights(
        #         samples.weights, highlight_dict=last_obs_w, save_path="data/samples.png"
        #     )
        return self._samples[last_name]

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    def _build_normalizer(self, task, task_name):
        """For new tasks, need to build normalizer."""
        assert self._normalizer_fn is not None, "Need to set random seed"
        norm_ws = self._normalizer_fn()
        state = self._get_init_state(task)
        normalizer = Particles(
            self._rng_key, self._env_fn, self._controller, self._runner, norm_ws
        )
        norm_feats = normalizer.get_features(
            task, task_name, desc="Collecting Normalizer Features"
        )
        return normalizer

    def _build_kernel(self, beta):
        """Likelihood for observed data, used as `PGM._kernel`.

        Builds IRD-specific kernel, which takes specialized
        arguments: `init_state`, `norm_feats_sums`, `user_feats_sums`.

        Args:
            prior_w (dict): sampled from prior
            user_w (dict): user-specified reward weight
            tasks (array): environment init state
            norm_feats_sum(array): samples used to normalize likelihood in
                inverse reward design problem. Sampled before running `pgm.sample`.

        TODO:
            * `user_ws` (represented as `obs` in generic sampling class) is not used in
            kernel, causing some slight confusion

        """

        def likelihood_fn(user_ws, sample_w, norm_feats_sums, user_feats_sums):
            """Main likelihood logic.

            Runs `p(w_obs | w)`

            """
            assert self._prior_log_prob is not None, "need to set random seed"

            log_probs = []
            # Iterate over list of previous tasks
            for n_feats_sum, u_feats_sum in zip(norm_feats_sums, user_feats_sums):
                for val in u_feats_sum.values():
                    assert len(val) == 1, "Only can take 1 user sample"
                sample_costs = multiply_dict_by_keys(sample_w, u_feats_sum)
                ## Numerator
                sample_rew = -beta * np.sum(list(sample_costs.values()))
                ## Normalizing constant
                normal_costs = multiply_dict_by_keys(sample_w, n_feats_sum)
                normal_rews = -beta * np.sum(list(normal_costs.values()), axis=0)
                sum_normal_rew = -np.log(len(normal_rews)) + logsumexp(normal_rews)

                if self._debug_true_w:
                    true_costs = multiply_dict_by_keys(
                        self._designer.true_w, u_feats_sum
                    )
                    true_rew = -beta * np.sum(list(true_costs.values()))
                    true_normal_costs = multiply_dict_by_keys(
                        self._designer.true_w, n_feats_sum
                    )
                    true_normal_rews = -beta * np.sum(
                        list(true_normal_costs.values()), axis=0
                    )
                    true_sum_normal_rew = -np.log(len(true_normal_rews)) + logsumexp(
                        true_normal_rews
                    )
                    print(
                        f"Sample rew {sample_rew - sum_normal_rew:.3f} true rew {true_rew - true_sum_normal_rew:.3f}"
                    )
                    # import pdb; pdb.set_trace()

                log_probs.append(sample_rew - sum_normal_rew)
            log_probs.append(self._prior_log_prob(sample_w))
            # if len(log_probs) > 2:
            #     print([f"{p:.3f}" for p in log_probs])
            return sum(log_probs)

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
        env_fn,
        controller,
        runner,
        beta,
        true_w,
        prior_log_prob_fn,
        proposal_fn,
        sample_method,
        sampler_args,
        use_true_w=False,
    ):
        self._rng_key = rng_key
        self._env_fn = env_fn
        self._env = env_fn()
        self._controller = controller
        self._runner = runner
        self._true_w = true_w
        self._use_true_w = use_true_w
        # Prior function
        self._prior_raw_fn = prior_log_prob_fn
        self._prior_log_prob = None
        # Sampling kernel
        kernel = self._build_kernel(beta)
        self._samples = {}
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sampler_args)

    def sample(self, task, task_name, verbose=True):
        """Sample 1 set of weights from b(design_w) given true_w"""
        if task_name not in self._samples.keys():
            print("Sampling Designer")
            if self._use_true_w:
                sample_ws = [self._true_w]
            else:
                sample_ws = self._sampler.sample(self._true_w, task=task)
            samples = Particles(
                self._rng_key, self._env_fn, self._controller, self._runner, sample_ws
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
        self._sampler.update_key(rng_key)
        self._prior_log_prob = seed(self._prior_raw_fn, rng_key)

    def _build_kernel(self, beta):
        def likelihood_fn(true_w, sample_w, task):
            assert self._prior_log_prob is not None, "Need to set random seed"
            state = self._get_init_state(task)
            actions = self._controller(state, weights=sample_w)
            _, cost, info = self._runner(state, actions, weights=true_w)
            rew = -1 * cost
            log_prob = beta * rew
            log_prob += self._prior_log_prob(sample_w)
            return log_prob

        return likelihood_fn
