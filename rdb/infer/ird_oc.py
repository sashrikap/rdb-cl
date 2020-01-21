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
import json
import os
import jax
import jax.numpy as np
from numpyro.handlers import scale, condition, seed
from rdb.infer.utils import random_choice, logsumexp, get_init_state
from rdb.optim.utils import multiply_dict_by_keys
from rdb.visualize.plot import plot_weights
from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from rdb.infer.algos import *
from time import time

import pprint

pp = pprint.PrettyPrinter(indent=4)


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Methods:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(
        self, rng_key, kernel, proposal_fn, sample_method="mh", sample_args={}
    ):
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
        true_w (dict):
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
        env_id,
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
        num_prior_tasks=0,
        use_true_w=False,
        interactive_mode=True,
        interactive_name="Default",
        debug_true_w=False,
        test_mode=False,  # Skip _cache_task part if true
    ):
        self._rng_key = rng_key
        # Environment settings
        self._env_id = env_id
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
        self._debug_true_w = debug_true_w
        self._test_mode = test_mode
        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sample_args)
        # assume designer uses the same beta, prior and prior proposal
        self._interactive_mode = interactive_mode
        if not interactive_mode:
            truth = self.create_particles([true_w])
            self._designer = Designer(
                rng_key,
                self.env_fn,
                self._controller,
                self._runner,
                beta,
                truth,
                prior_log_prob_fn,
                proposal_fn,
                sample_method,
                designer_args,
                use_true_w=use_true_w,
                num_prior_tasks=num_prior_tasks,
            )
        else:
            # Interactive Mode
            self._designer = DesignerInteractive(
                rng_key, self.env_fn, self._controller, self._runner, interactive_name
            )

    @property
    def env_id(self):
        return self._env_id

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

    @property
    def interactive_mode(self):
        return self._interactive_mode

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
        return self._designer.simulate(task, task_name)

    def sample(self, tasks, task_names, obs, verbose=True):
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
        samples = self.create_particles(sample_ws, self._test_mode)
        self._samples[last_name] = samples
        return self._samples[last_name]

    def create_particles(self, weights, test_mode=False):
        return Particles(
            self._rng_key,
            self._env_fn,
            self._controller,
            self._runner,
            weights,
            test_mode=test_mode,
        )

    def _build_normalizer(self, task, task_name):
        """For new tasks, need to build normalizer."""
        assert self._normalizer_fn is not None, "Need to set random seed"
        norm_ws = self._normalizer_fn()
        state = get_init_state(self.env, task)
        normalizer = self.create_particles(norm_ws, test_mode=False)
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
    """Simulated designer module.

    Given true weights, sample MAP design weights.

    Note:
        * Currently assumes same beta, proposal as IRD module, although
          doesn't have to be.

    Args:
        truth (Particles): true weight particles

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller,
        runner,
        beta,
        truth,
        prior_log_prob_fn,
        proposal_fn,
        sample_method="mh",
        sampler_args={},
        use_true_w=False,
        num_prior_tasks=0,
    ):
        self._rng_key = rng_key
        self._env_fn = env_fn
        self._env = env_fn()
        self._controller = controller
        self._runner = runner
        self._truth = truth
        if self._truth is not None:
            self._true_w = truth.weights[0]
        else:
            self._true_w = None
        self._use_true_w = use_true_w
        # Prior function
        self._prior_raw_fn = prior_log_prob_fn
        self._prior_log_prob = None
        # Sampling kernel
        kernel = self._build_kernel(beta)
        self._samples = {}
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sampler_args)
        # Designer Prior
        self._random_choice = None
        self._prior_tasks = []
        self._num_prior_tasks = num_prior_tasks

    def sample(self, task, task_name, verbose=True):
        """Sample 1 set of weights from b(design_w) given true_w"""
        if task_name not in self._samples.keys():
            print(f"Sampling Designer (prior={len(self._prior_tasks)})")
            if self._use_true_w:
                sample_ws = [self._true_w]
            else:
                sample_ws = self._sampler.sample(self._true_w, task=task)
            samples = Particles(
                self._rng_key, self._env_fn, self._controller, self._runner, sample_ws
            )
            self._samples[task_name] = samples
        return self._samples[task_name]

    def simulate(self, task, task_name):
        """Give 1 sample"""
        particles = self.sample(task, task_name)
        return particles.subsample(1)

    def reset_prior_tasks(self):
        assert (
            self._random_choice is not None
        ), "Need to initialize Designer with random seed"
        assert self._num_prior_tasks < len(self._env.all_tasks)
        tasks = self._random_choice(self._env.all_tasks, self._num_prior_tasks)
        self._prior_tasks = tasks

    @property
    def prior_tasks(self):
        return self._prior_tasks

    @prior_tasks.setter
    def prior_tasks(self, tasks):
        """ Modifies Underlying Designer Prior. Use very carefully."""
        self._prior_tasks = tasks

    @property
    def true_w(self):
        return self._true_w

    @property
    def truth(self):
        return self._truth

    @property
    def env(self):
        return self._env

    def update_key(self, rng_key):
        super().update_key(rng_key)
        self._truth.update_key(rng_key)
        self._sampler.update_key(rng_key)
        self._prior_log_prob = seed(self._prior_raw_fn, rng_key)
        self._random_choice = seed(random_choice, rng_key)
        self.reset_prior_tasks()

    def _build_kernel(self, beta):
        def likelihood_fn(true_w, sample_w, task):
            assert self._prior_log_prob is not None, "Need to set random seed"
            state = get_init_state(self.env, task)
            actions = self._controller(state, weights=sample_w)
            _, cost, info = self._runner(state, actions, weights=true_w)
            rew = -1 * cost
            log_prob = beta * rew
            log_prob += self._prior_log_prob(sample_w)
            return log_prob

        return likelihood_fn


class DesignerInteractive(Designer):
    """Interactive Designer used in Jupyter Notebook interactive mode.
    """

    def __init__(self, rng_key, env_fn, controller, runner, name="Default"):
        super().__init__(
            rng_key=rng_key,
            env_fn=env_fn,
            controller=controller,
            runner=runner,
            beta=None,
            truth=None,
            prior_log_prob_fn=None,
            proposal_fn=None,
            sample_method=None,
        )
        self._name = name
        self._savedir = self.init_savedir()
        self._user_inputs = []

    def run_from_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    def init_savedir(self):
        """Create empty directory to dump interactive videos.
        """
        # By default, this is run from notebook
        assert self.run_from_ipython()
        # nb directory: "/Users/jerry/Dropbox/Projects/SafeRew/rdb/examples/notebook"
        savedir = f"../../data/interactive/{self._name}"
        os.makedirs(savedir, exist_ok=True)
        return savedir

    def sample(self, task, task_name, verbose=True):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def simulate(self, task, task_name):
        """Interactively query weight input.

        In Jupyter Notebook:
            * Display a task visualization.
            * Show previous weights, current MAP belief
            * Let user play around with different weights
            * Let user feed in a selected weight.

        Todo:
            * Display all other not-selected candidates.
        """
        self.env.set_task(task)
        init_state = get_init_state(self.env, task)
        # Visualize task
        image_path = f"{self._savedir}/user_trial_0_task_{str(task)}.png"
        self._runner.nb_show_thumbnail(init_state, image_path, clear=False)
        # Query user input
        while True:
            print(f"Current task: {str(task)}")
            user_in = input("Type in weights or Y to accept last")
            if user_in == "Y":
                if len(self._user_inputs) == 0:
                    print("Need at least one input")
                    continue
                else:
                    break
            try:
                user_in_w = json.loads(user_in)
                self._user_inputs.append(user_in_w)
                # Visualize trajectory
                acs = self._controller(init_state, weights=user_in_w)
                num_weights = len(self._user_inputs)
                video_path = (
                    f"{self._savedir}/user_trial_{num_weights}_task_{str(task)}.mp4"
                )
                print("Received Weights")
                pp.pprint(user_in_w)
                self._runner.nb_show_mp4(init_state, acs, path=video_path, clear=False)
            except Exception as e:
                print(e)
                print("Invalid input.")
                continue
        return Particles(
            self._rng_key,
            self._env_fn,
            self._controller,
            self._runner,
            [self._user_inputs[-1]],
        )

    def reset_prior_tasks(self):
        raise NotImplementedError

    @property
    def prior_tasks(self):
        raise NotImplementedError

    @prior_tasks.setter
    def prior_tasks(self, tasks):
        raise NotImplementedError

    @property
    def true_w(self):
        return None

    @property
    def truth(self):
        return None

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def _build_sampler(self, kernel, proposal_fn, sample_method, sample_args):
        """Dummy sampler."""
        pass

    def _build_kernel(self, beta):
        """Dummy kernel."""
        pass
