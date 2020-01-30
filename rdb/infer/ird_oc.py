"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] General PGM inference API
    [2] Reward Design with Divide and Conquer
    [3] Different ways (max_norm/sample/hybrid) to approximate IRD normalizer
    [4] Designer Class
    [5] Informed Designer with prior tasks, see `rdb/exps/designer_prior.py`

Note:
    * We use `init_state/task` interchangably. In drive2d
     `env.set_task(init_state)` specifies a task to run.

"""

import numpyro
import pprint
import copy
import json
import os
import jax
import jax.numpy as np
import numpy as onp
from rdb.optim.utils import multiply_dict_by_keys, append_dict_by_keys
from numpyro.handlers import scale, condition, seed
from rdb.infer.utils import random_choice, logsumexp
from rdb.infer.particles import Particles
from tqdm.auto import tqdm, trange
from rdb.exps.utils import *
from os.path import join
from rdb.infer.algos import *
from rdb.infer.utils import *
from time import time

pp = pprint.PrettyPrinter(indent=4)


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Methods:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel, proposal, sample_method="mh", sample_args={}):
        self._rng_key = rng_key
        self._sampler = self._build_sampler(
            kernel, proposal, sample_method, sample_args
        )

    def update_key(self, rng_key):
        self._rng_key = rng_key
        # self._sampler.update_key(rng_key)

    def _build_sampler(self, kernel, proposal, sample_method, sample_args):
        if sample_method == "mh":
            return MetropolisHasting(
                self._rng_key, kernel, proposal=proposal, **sample_args
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
        prior (Prior): sampling prior
        num_normalizers (int): fixed number of normalizers
        true_w (dict):
        use_true_w (bool): debug designer with true w

    Methods:
        sample (fn): sample b(w) given obs_w
        update (fn): incorporate user's obs_w
            next likelihood `p(w | obs_w1) p(w | obs_w2)...`
        infer (fn): infer p`(w* | obs)`

    Example:
        * see rdb/examples/run_active.py

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
        prior,
        proposal,
        ## Weight parameters
        num_normalizers=-1,
        normalized_key=None,
        ## Sampling
        sample_method="mh",
        sample_args={},  # "num_warmups": 100, "num_samples": 200
        ## Designer
        designer_proposal=None,
        designer_args={},
        num_prior_tasks=0,
        use_true_w=False,
        ## Parameter for histogram
        weight_params={},
        interactive_mode=False,
        interactive_name="Default",
        ## Saving options
        save_root="data",
        exp_name="active_ird_exp1",
        ## Debugging options
        debug_true_w=False,
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
        self._prior = prior
        self._num_normalizers = num_normalizers
        self._normalized_key = normalized_key
        ## Caching normalizers, samples and features
        self._normalizer = None
        self._user_actions = {}
        self._user_feats = {}
        self._debug_true_w = debug_true_w
        self._weight_params = weight_params
        self._kernel = self._build_kernel(beta)
        ## Saving
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        super().__init__(rng_key, self._kernel, proposal, sample_method, sample_args)
        # assume designer uses the same beta, prior and prior proposal
        self._interactive_mode = interactive_mode
        if interactive_mode:
            # Interactive Mode
            self._designer = DesignerInteractive(
                rng_key=rng_key,
                env_fn=self.env_fn,
                controller=self._controller,
                runner=self._runner,
                name=interactive_name,
                normalized_key=self._normalized_key,
            )
        else:
            # Normal Mode
            assert designer_proposal is not None, "Must specify designer proposal"
            self._designer = Designer(
                rng_key=rng_key,
                env_fn=self.env_fn,
                controller=self._controller,
                runner=self._runner,
                beta=beta,
                truth=self.create_particles([true_w], save_name="true_w"),
                prior=prior,
                proposal=designer_proposal,
                sample_method=sample_method,
                sampler_args=designer_args,
                use_true_w=use_true_w,
                weight_params=weight_params,
                num_prior_tasks=num_prior_tasks,
                normalized_key=self._normalized_key,
                save_root=save_root,
                exp_name=exp_name,
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
        self._prior.update_key(rng_key)

    def simulate_designer(self, task, task_name, save_name):
        """Sample one w from b(w) on task

        Args:
            path (str): path to save png
            params (dict): visualization params
            save_name (str)

        """
        return self._designer.simulate(task, task_name, save_name=save_name)

    def _get_chain_viz(self, save_name, num_plots=10):
        """Visualize multiple MCMC chains to check convergence.
        """

        def fn(samples, accepts):
            fig_dir = f"{self._save_dir}/mcmc"
            visualize_chains(
                samples, accepts, num_plots=num_plots, fig_dir=fig_dir, title=save_name
            )

        return fn

    def sample(self, tasks, task_names, obs, save_name, verbose=True, mode="hybrid"):
        """Sample b(w) for true weights given obs.weights.

        Args:
            tasks (list): list of tasks so far
            task_names (list): list of task names so far
            obs (list): list of observation particles, each for 1 task
            mode (string): how to estimate normalizer (doubly-intractable)
                `sample`: use random samples for normalizer;
                `max_norm`: use max trajectory as normalizer;
                `hybrid`: use random samples mixed with max trajectory

        Note:
            * Samples are cached by the last task name to date (`task_names[-1]`)

        TODO:
            * currently `tasks`, `task_names`, `obs` are ugly lists

        """
        assert (
            len(tasks) == len(task_names) == len(obs)
        ), "Tasks and observations mismatch"
        assert mode in [
            "hybrid",
            "max_norm",
            "sample",
        ], "Must specify IRD sampling mode"

        all_obs_ws = []
        all_user_acs = []
        all_init_states = []
        all_user_feats_sum = []
        all_norm_feats_sum = []
        for task_i, name_i, obs_i in zip(tasks, task_names, obs):
            init_state = self.env.get_init_state(task_i)
            all_init_states.append(init_state)
            all_obs_ws.append(obs_i.weights[0])
            all_user_feats_sum.append(obs_i.get_features_sum(task_i, name_i))
            all_user_acs.append(obs_i.get_actions(task_i, name_i))

        if mode == "max_norm":
            all_norm_feats_sum = [None for _ in range(len(tasks))]
        elif mode == "sample" or mode == "hybrid":
            if self._normalizer is None:
                self._normalizer = self._build_normalizer()
            for task_i, name_i, obs_i in zip(tasks, task_names, obs):
                all_norm_feats_sum.append(
                    self._normalizer.get_features_sum(
                        task_i, name_i, "Computing Normalizer Features"
                    )
                )
        else:
            raise NotImplementedError

        last_obs_w = obs[-1].weights[0]
        last_name = task_names[-1]

        print("Sampling IRD")
        sample_ws = self._sampler.sample(
            obs=all_obs_ws,  # all obs so far
            init_state=last_obs_w,  # initialize with last obs
            chain_viz=self._get_chain_viz(save_name=save_name),
            user_feats_sums=all_user_feats_sum,
            norm_feats_sums=all_norm_feats_sum,
            all_init_states=all_init_states,
            all_user_acs=all_user_acs,
            verbose=verbose,
            mode=mode,
            name=save_name,
        )
        samples = self.create_particles(sample_ws, save_name=save_name)
        return samples

    def create_particles(self, weights, save_name):
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=self._controller,
            runner=self._runner,
            sample_ws=weights,
            save_name=save_name,
            weight_params=self._weight_params,
            fig_dir=f"{self._save_dir}/plots",
            save_dir=f"{self._save_dir}/save",
            env=self._env,
        )

    def _build_normalizer(self):
        """Build sampling-based normalizer by randomly sampling weights."""
        norm_ws = self._prior.sample(self._num_normalizers)
        normalizer = self.create_particles(norm_ws, save_name="normalizer")
        return normalizer

    def _build_kernel(self, beta):
        """Likelihood for observed data, used as `PGM._kernel`.

        Builds IRD-specific kernel, which takes specialized
        arguments: `init_state`, `norm_feats_sums`, `user_feats_sums`.

        Args:
            prior_w (dict): sampled from prior
            user_w (dict): user-specified reward weight
            tasks (array): environment init state
            norm_feats_sum(array): samples used to normalize . likelihood in
                inverse reward design problem. Sampled before running `pgm.sample`.
            user_feats_sums (list)
            all_init_states (list)
            all_user_acs (list)

        TODO:
            * `user_ws` (represented as `obs` in generic sampling class) is not used in
            kernel, causing some slight confusion

        """

        def likelihood_fn(
            user_ws,
            sample_w,
            norm_feats_sums,
            user_feats_sums,
            all_init_states,
            all_user_acs,
            mode="hybrid",
        ):
            """Main likelihood logic.

            Runs `p(w_obs | w)`

            """
            log_probs = []
            # Iterate over list of previous tasks
            for n_feats_sum, u_feats_sum, init_state, user_acs in zip(
                norm_feats_sums, user_feats_sums, all_init_states, all_user_acs
            ):
                for val in u_feats_sum.values():
                    assert len(val) == 1, "Only can take 1 user sample"
                sample_costs = multiply_dict_by_keys(sample_w, u_feats_sum)
                ## Numerator
                sample_rew = -beta * np.sum(list(sample_costs.values()))

                ## Estimating Normalizing constant
                if mode == "max_norm":
                    # Use max trajectory to replace normalizer
                    # Important to initialize from observation actions
                    assert n_feats_sum is None
                    sample_acs = self._controller(
                        init_state, us0=user_acs, weights=sample_w
                    )
                    _, sample_costs, info = self._runner(
                        init_state, sample_acs, weights=sample_w
                    )
                    sum_normal_rew = -beta * sample_costs

                elif mode == "sample":
                    # Use weight samples of approximate normalizer
                    normal_costs = multiply_dict_by_keys(sample_w, n_feats_sum)
                    normal_rews = -beta * np.sum(list(normal_costs.values()), axis=0)
                    sum_normal_rew = -np.log(len(normal_rews)) + logsumexp(normal_rews)

                elif mode == "hybrid":
                    # Use both max trajectory and weight samples to approximate normalizer
                    sample_acs = self._controller(
                        init_state, us0=user_acs, weights=sample_w
                    )
                    _, sample_cost, sample_info = self._runner(
                        init_state, sample_acs, weights=sample_w
                    )
                    # print("Sample cost", sample_cost)
                    n_feats_sum = append_dict_by_keys(
                        n_feats_sum, sample_info["feats_sum"]
                    )
                    normal_costs = multiply_dict_by_keys(sample_w, n_feats_sum)
                    normal_rews = -beta * np.sum(list(normal_costs.values()), axis=0)
                    sum_normal_rew = -np.log(len(normal_rews)) + logsumexp(normal_rews)
                else:
                    raise NotImplementedError

                if self._debug_true_w:
                    # if True:
                    print(
                        f"sample rew {sample_rew:.3f} normal costs {sum_normal_rew:.3f} diff {sample_rew - sum_normal_rew:.3f}"
                    )

                log_probs.append(sample_rew - sum_normal_rew)
            log_probs.append(self._prior.log_prob(sample_w))
            return sum(log_probs)

        def _kernel(user_ws, sample_w, **kwargs):
            """Vectorized"""
            if isinstance(sample_w, dict):
                return likelihood_fn(user_ws, sample_w, **kwargs)
            else:
                probs = []
                for uw, sw in zip(user_ws, sample_w):
                    probs.append(likelihood_fn(us, sw, **kwargs))
                return onp.array(probs)

        return _kernel


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
        prior,
        proposal,
        sample_method="mh",
        sampler_args={},
        save_root="",
        exp_name="",
        use_true_w=False,
        weight_params={},
        num_prior_tasks=0,
        normalized_key=None,
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
        self._normalized_key = normalized_key
        # Sampling Prior and kernel
        self._prior = prior
        self._kernel = self._build_kernel(beta)
        # Cache
        self._task_samples = {}
        super().__init__(rng_key, self._kernel, proposal, sample_method, sampler_args)
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        # Designer Prior
        self._random_choice = None
        self._prior_tasks = []
        self._weight_params = weight_params
        self._num_prior_tasks = num_prior_tasks

    def _get_chain_viz(self, save_name, num_plots=5):
        """Visualize multiple MCMC chains to check convergence.
        """

        def fn(samples, accepts):
            fig_dir = f"{self._save_dir}/mcmc"
            visualize_chains(
                samples, accepts, num_plots=num_plots, fig_dir=fig_dir, title=save_name
            )

        return fn

    def create_particles(self, weights, save_name):
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=self._controller,
            runner=self._runner,
            sample_ws=weights,
            save_name=save_name,
            weight_params=self._weight_params,
            fig_dir=f"{self._save_dir}/plots",
            save_dir=f"{self._save_dir}/save",
            env=self._env,
        )

    def sample(self, task, task_name, save_name):
        """Sample 1 set of weights from b(design_w) given true_w

        Args:
            task: environment task
            task_name (str):
            save_name (save_name): name for saving parameters and visualizations

        """
        if task_name not in self._task_samples.keys():
            print(f"Sampling Designer (prior={len(self._prior_tasks)}): {save_name}")
            if self._use_true_w:
                sample_ws = [self._true_w]
            else:
                sample_ws = self._sampler.sample(
                    self._true_w,
                    task=task,
                    chain_viz=self._get_chain_viz(save_name=save_name),
                    name=save_name,
                )
            samples = Particles(
                rng_key=self._rng_key,
                env_fn=self._env_fn,
                controller=self._controller,
                runner=self._runner,
                weight_params=self._weight_params,
                sample_ws=sample_ws,
                save_name=save_name,
                fig_dir=f"{self._save_dir}/plots",
                save_dir=f"{self._save_dir}/save",
                env=self._env,
            )
            self._task_samples[task_name] = samples
        return self._task_samples[task_name]

    def simulate(self, task, task_name, save_name):
        """Give 1 sample

        Args:
            save_name (str): for later saving output

        """
        particles = self.sample(task, task_name, save_name=save_name)
        particles.visualize(true_w=self.true_w, obs_w=None)
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
        self._prior.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        self.reset_prior_tasks()

    def _build_kernel(self, beta):
        def likelihood_fn(true_w, sample_w, task):
            assert self._prior_tasks is not None, "Need >=0 prior tasks."
            all_tasks = self._prior_tasks + [task]
            log_prob = 0.0
            for task_i in all_tasks:
                state = self.env.get_init_state(task)
                actions = self._controller(state, weights=sample_w)
                _, cost, info = self._runner(state, actions, weights=true_w)
                rew = -1 * cost
                log_prob += beta * rew
            log_prob += self._prior.log_prob(sample_w)
            # if True:
            if False:
                print(f"Designer prob {log_prob:.3f}", actions.mean())
            return log_prob

        def _kernel(true_w, sample_w, **kwargs):
            """Vectorized"""
            if isinstance(sample_w, dict):
                return likelihood_fn(true_w, sample_w, **kwargs)
            else:
                probs = []
                for tw, w in zip(true_w, sample_w):
                    probs.append(likelihood_fn(tw, w, **kwargs))
                return onp.array(probs)

        return _kernel


class DesignerInteractive(Designer):
    """Interactive Designer used in Jupyter Notebook interactive mode.

    Args:
        normalized_key (str): used to normalize user-input, e.g. keep "dist_cars" weight 1.0.

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller,
        runner,
        name="Default",
        save_dir="",
        normalized_key=None,
    ):
        super().__init__(
            rng_key=rng_key,
            env_fn=env_fn,
            controller=controller,
            runner=runner,
            beta=None,
            truth=None,
            prior=None,
            proposal=None,
            sample_method=None,
            save_dir=save_dir,
            exp_name="interactive",
        )
        self._name = name
        self._save_dir = f"{self._save_dir}/{self._exp_name}/{save_name}"
        os.makedirs(self._save_dir)
        self._user_inputs = []
        self._normalized_key = normalized_key

    def sample(self, task, task_name, verbose=True, itr=0):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def simulate(self, task, task_name, save_name):
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
        init_state = self.env.get_init_state(task)
        # Visualize task
        image_path = (
            f"{self._save_dir}/key_{self._rng_key}_user_trial_0_task_{str(task)}.png"
        )
        self._runner.nb_show_thumbnail(init_state, image_path, clear=False)
        # Query user input
        while True:
            print(f"Current task: {str(task)}")
            user_in = input("Type in weights or to accept (Y) last: ")
            if user_in == "Y" or user_in == "y":
                if len(self._user_inputs) == 0:
                    print("Need at least one input")
                    continue
                else:
                    break
            try:
                user_in_w = json.loads(user_in)
                user_in_w = normalize_weights(user_in_w, self._normalized_key)
                self._user_inputs.append(user_in_w)
                # Visualize trajectory
                acs = self._controller(init_state, weights=user_in_w)
                num_weights = len(self._user_inputs)
                video_path = f"{self._save_dir}/key_{self._rng_key}_user_trial_{num_weights}_task_{str(task)}.mp4"
                print("Received Weights")
                pp.pprint(user_in_w)
                self._runner.nb_show_mp4(init_state, acs, path=video_path, clear=False)
            except Exception as e:
                print(e)
                print("Invalid input.")
                continue
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=self._controller,
            runner=self._runner,
            sample_ws=[self._user_inputs[-1]],
            save_name=f"designer_interactive_{self._name}",
            weight_params=self._weight_params,
            env=self._env,
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

    def _build_sampler(self, kernel, proposal, sample_method, sample_args):
        """Dummy sampler."""
        pass

    def _build_kernel(self, beta):
        """Dummy kernel."""
        pass
