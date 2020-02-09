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
from rdb.infer.designer import Designer, DesignerInteractive
from numpyro.handlers import scale, condition, seed
from rdb.infer.utils import random_choice, logsumexp
from rdb.infer.particles import Particles
from rdb.infer.pgm import PGM
from tqdm.auto import tqdm, trange
from rdb.infer.algos import *
from rdb.infer.utils import *
from rdb.exps.utils import *
from os.path import join
from time import time

pp = pprint.PrettyPrinter(indent=4)


class IRDOptimalControl(PGM):
    """Inverse Reward Design for Optimal Control.

    Given environment `env`, `planner`, user input w
    infer p(w* | w).

    Notes:
        * Bayesian Terminology: theta -> true w, obs -> proxy w
        * Provides caching functions to avoid costly feature calcultions
        * To reduce JIT recompile time, uses separate controller for (1) normalizers, (2) batch kernel, (3) samples and (4) designer

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
        normalized_key,
        num_normalizers=-1,
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
    ):
        self._rng_key = rng_key
        # Environment settings
        self._env_id = env_id
        self._env_fn = env_fn
        self._env = env_fn()
        self._designer_controller, self._designer_runner = controller_fn(self._env)
        self._sample_controller, self._sample_runner = controller_fn(self._env)
        self._norm_controller, self._norm_runner = controller_fn(self._env)
        self._batch_controller, self._batch_runner = controller_fn(self._env)
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
        self._weight_params = weight_params
        self._kernel = self._build_kernel()
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
                controller=self._designer_controller,
                runner=self._designer_runner,
                name=interactive_name,
                normalized_key=self._normalized_key,
            )
        else:
            # Normal Mode
            assert designer_proposal is not None, "Must specify designer proposal"
            self._designer = Designer(
                rng_key=rng_key,
                env_fn=self.env_fn,
                controller=self._designer_controller,
                runner=self._designer_runner,
                beta=beta,
                true_w=true_w,
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
        """Build sampling-based normalizer by randomly sampling weights."""
        norm_ws = self._prior.sample(self._num_normalizers)
        self._normalizer = self.create_particles(
            norm_ws,
            save_name="ird_normalizer",
            runner=self._norm_runner,
            controller=self._norm_controller,
        )

    def simulate_designer(self, task, task_name, save_name):
        """Sample one w from b(w) on task

        Args:
            path (str): path to save png
            params (dict): visualization params
            save_name (str)

        """
        return self._designer.simulate(task, task_name, save_name=save_name)

    def sample(self, tasks, task_names, obs, save_name, verbose=True):
        """Sample b(w) for true weights given obs.weights.

        Args:
            tasks (list): list of tasks so far
            task_names (list): list of task names so far
            obs (list): list of observation particles, each for 1 task

        Note:
            * Samples are cached by the last task name to date (`task_names[-1]`)
            * To approximate IRD doubly-intractible denominator:
                `sample`: use random samples for normalizer;
                `max_norm`: use max trajectory as normalizer;
                `hybrid`: use random samples mixed with max trajectory

        TODO:
            * currently `tasks`, `task_names`, `obs` are ugly lists

        """
        assert len(tasks) > 0, "Need >=1 tasks"
        assert (
            len(tasks) == len(task_names) == len(obs)
        ), "Tasks and observations mismatch"
        assert self._normalizer is not None

        init_states = self.env.get_init_states(tasks)
        obs_ws = []
        user_acs = []
        norm_feats = []
        for task_i, name_i, obs_i in zip(tasks, task_names, obs):
            obs_ws.append(obs_i.weights[0])
            user_acs.append(obs_i.get_actions(task_i, name_i)[0])
            norm_feats.append(self._normalizer.get_features_sum(task_i, name_i))

        print(f"Sampling IRD (obs={len(obs)}): {save_name}")

        last_obs_w = obs[-1].weights[0]
        #  shape (ntasks, T, acs_dim)
        user_acs = onp.array(user_acs)
        #  shape nfeats * (ntasks, n_normalizer)
        norm_feats = DictList(norm_feats)
        norm_feats = norm_feats.prepare(self._env.features_keys)
        sample_ws, info = self._sampler.sample(
            obs=obs_ws,  # all obs so far
            init_state=last_obs_w,  # initialize with last obs
            user_acs=user_acs,
            norm_feats=norm_feats,
            tasks=tasks,
            name=save_name,
        )
        num_samples = len(info["all_chains"][0])
        visualize_chains(
            chains=info["all_chains"],
            rates=info["rates"],
            fig_dir=f"{self._save_dir}/mcmc",
            title=f"seed_{str(self._rng_key)}_{save_name}_samples_{num_samples}",
            **self._weight_params,
        )
        sample_ws = DictList(sample_ws)
        samples = self.create_particles(
            sample_ws,
            save_name=save_name,
            runner=self._sample_runner,
            controller=self._sample_controller,
        )
        samples.visualize(true_w=self.designer.true_w, obs_w=last_obs_w)
        return samples

    def create_particles(self, weights, save_name, runner, controller):
        weights = DictList(weights)
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=controller,
            runner=runner,
            weights=weights,
            save_name=save_name,
            normalized_key=self._normalized_key,
            weight_params=self._weight_params,
            fig_dir=f"{self._save_dir}/plots",
            save_dir=f"{self._save_dir}/save",
            env=self._env,
        )

    def _cross(self, data_a, data_b, type_a, type_b):
        """To do compuation on each a for each b.

        Performs Cross product: (num_a,) x (num_b,) -> (num_a * num_b,).

        Output:
            batch_a (type_a): shape (num_a * num_b,)
            batch_b (type_b): shape (num_a * num_b,)

        """

        pairs = list(itertools.product(data_a, data_b))
        batch_a, batch_b = zip(*pairs)
        return type_a(batch_a), type_b(batch_b)

    def _build_kernel(self):
        """Forward likelihood for observed data, used as MCMC kernel.

        Finds `p(w_obs | w_true)`.

        Args:
            user_ws (DictList): designer-specified reward weight
                shape nfeats * (ntasks,)
            sample_ws (DictList: MCMC sample
                shape nfeats * (nchains,)
            user_acs (ndarray): user_ws' actions
                shape (ntasks, T, acs_dim)
            norm_feats (DictList): normalizer's features
                shape nfeats * (ntasks, n_normalizers)
            tasks (ndarray): tasks, (ntasks, task_dim)

        Output:
            log_probs (ndarray): log probability of sample_ws (nchains, 1)

        Note:
            * To stabilize MH sampling
                (1) average across tasks (instead of sum)
                (2) average across features costs (instead of sum)

        TODO:
            * `user_ws` (represented as `obs` in generic sampling class) is somewhat confusingly, not used in
            kernel. Better API design for algo.py

        """

        def likelihood_fn(
            user_ws,
            sample_ws,
            user_acs,
            norm_feats,
            tasks,
            normalize_across_keys=True,
            extend_norm=False,
            one_norm=True,
        ):

            assert len(user_ws) == len(norm_feats) == len(user_acs)  #  length (ntasks)

            prior_probs = self._prior.log_prob(sample_ws)
            nnorms = norm_feats.shape[1]
            nfeats = sample_ws.num_keys
            nchain = len(sample_ws)
            ntasks = len(user_ws)

            sample_ws = sample_ws.prepare(self._env.features_keys)
            if normalize_across_keys:
                sample_ws = sample_ws.normalize_across_keys()

            ## To calculuate likelihood on each sample each task
            ## Cross product: (nchain,) x (ntasks,) -> (nchain * ntasks,)
            #  shape batch_sample_ws nfeats * (nchain * ntasks)
            #  shape batch_tasks (nchain * ntasks, task_dim)
            batch_sample_ws, batch_tasks = self._cross(
                sample_ws, tasks, DictList, onp.array
            )
            #  shape (nchain * ntasks, T, acs_dim)
            _, batch_user_acs = self._cross(sample_ws, user_acs, DictList, onp.array)
            #  shape nfeats * (nchain * ntasks, n_normalizers)
            _, batch_norm_feats = self._cross(sample_ws, norm_feats, DictList, DictList)
            #  shape (nchain * ntasks, state_dim)
            batch_init_states = self.env.get_init_states(batch_tasks)
            if extend_norm:
                ## Compute max rews for sample_ws
                #  shape (T, nchain * ntasks, acs_dim)
                batch_sample_acs = self._batch_controller(
                    batch_init_states, batch_sample_ws, us0=batch_user_acs
                )
                _, _, batch_info = self._batch_runner(
                    batch_init_states, batch_sample_acs, weights=batch_sample_ws
                )
                #  shape nfeats * (nchain * ntasks, n_normalizers + 1)
                batch_norm_feats = batch_norm_feats.concat(
                    batch_info["feats_sum"].expand_dims(axis=1), axis=1
                )
            elif one_norm:
                ## Compute max rews for sample_ws
                #  shape (T, nchain * ntasks, acs_dim)
                batch_sample_acs = self._batch_controller(
                    batch_init_states, batch_sample_ws, us0=batch_user_acs
                )
                _, _, batch_info = self._batch_runner(
                    batch_init_states, batch_sample_acs, weights=batch_sample_ws
                )
                #  shape nfeats * (nchain * ntasks, n_normalizers + 1)
                batch_norm_feats = batch_info["feats_sum"].expand_dims(axis=1)

            #  shape (nchain * ntasks, n_normalizers + 1)
            batch_norm_costs = (
                (batch_sample_ws.expand_dims(axis=1) * batch_norm_feats)
                .onp_array()
                .sum(axis=0)
            ) / nfeats
            #  shape (nchain * ntasks, )
            batch_norm_rews = -onp.log(nnorms + 1) + logsumexp(
                -self._beta * batch_norm_costs, axis=1
            )
            ## Denominator: Average across tasks (nchain * ntasks,) -> (nchain,)
            norm_rews = batch_norm_rews.reshape((nchain, ntasks)).mean(axis=1)

            #  shape (nchain * ntasks, )
            _, batch_user_costs, _ = self._batch_runner(
                batch_init_states, batch_user_acs, weights=batch_sample_ws
            )
            ## Numerator: Average across tasks (nchain * ntasks,) -> (nchain,)
            batch_user_costs /= nfeats

            user_rews = -self._beta * batch_user_costs.reshape((nchain, ntasks)).mean(
                axis=1
            )

            ## Debug
            # if True:
            if False:
                diff = onp.array(batch_norm_costs[0, :] - batch_user_costs[0])
                print(
                    f"Diff mean {diff.mean():03f} median {onp.median(diff):03f} std {diff.std():.03f}"
                )
                print(
                    f"sample rew {user_rews} normal costs {norm_rews} diff {user_rews - norm_rews}"
                )

            log_probs = user_rews - norm_rews
            log_probs += prior_probs
            return log_probs

        return likelihood_fn
