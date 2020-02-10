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
        designer_num_normalizers=-1,
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
        self._build_controllers(controller_fn)
        self._eval_server = eval_server
        # Rationality
        self._beta = beta
        # Sampling functions
        self._prior = prior
        self._num_normalizers = num_normalizers
        self._normalizer = None
        self._normalized_key = normalized_key
        ## Caching normalizers, samples and features
        self._user_actions = {}
        self._user_feats = {}
        self._weight_params = weight_params
        self._kernel = self._build_kernel()
        ## Saving
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        super().__init__(
            rng_key, self._kernel, prior, proposal, sample_method, sample_args
        )
        # assume designer uses the same beta, prior and prior proposal
        self._interactive_mode = interactive_mode
        if interactive_mode:
            # Interactive Mode
            self._designer = DesignerInteractive(
                rng_key=rng_key,
                env_fn=self.env_fn,
                controller_fn=controller_fn,
                name=interactive_name,
                normalized_key=self._normalized_key,
            )
        else:
            # Normal Mode
            assert designer_proposal is not None, "Must specify designer proposal"
            self._designer = Designer(
                rng_key=rng_key,
                env_fn=self.env_fn,
                controller_fn=controller_fn,
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
                num_normalizers=designer_num_normalizers,
                save_root=save_root,
                exp_name=exp_name,
            )

    def _build_controllers(self, controller_fn):
        self._sample_controller, self._sample_runner = controller_fn(
            self._env, "IRD Sample"
        )
        self._normal_controller, self._normal_runner = controller_fn(
            self._env, "IRD Normal"
        )
        self._batch_controller, self._batch_runner = controller_fn(
            self._env, "IRD Batch"
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
        normal_ws = self._prior(self._num_normalizers)
        self._normalizer = self.create_particles(
            normal_ws,
            save_name="ird_normalizer",
            runner=self._normal_runner,
            controller=self._normal_controller,
        )

    def simulate_designer(self, task, save_name):
        """Sample one w from b(w) on task

        Args:
            path (str): path to save png
            params (dict): visualization params
            save_name (str)

        """
        return self._designer.simulate(task, save_name=save_name)

    def sample(self, tasks, obs, save_name, verbose=True):
        """Sample b(w) for true weights given obs.weights.

        Args:
            tasks (list): list of tasks so far
            obs (list): list of observation particles, each for 1 task

        Note:
            * To approximate IRD doubly-intractible denominator:
                `sample`: use random samples for normalizer;
                `max_norm`: use max trajectory as normalizer;
                `hybrid`: use random samples mixed with max trajectory

        TODO:
            * currently `tasks`, `obs` are ugly lists

        """
        ntasks = len(tasks)
        assert len(tasks) > 0, "Need >=1 tasks"
        assert len(tasks) == len(obs), "Tasks and observations mismatch"
        assert self._normalizer is not None

        print(f"Sampling IRD (obs={len(obs)}): {save_name}")

        ## Preempt computations
        self._normalizer.compute_tasks(tasks, vectorize=False)
        #  shape nfeats * (ntasks,)
        obs_ws = DictList([ob.weights for ob in obs]).squeeze(axis=1)
        assert obs_ws.shape == (ntasks,)

        sample_ws, info = self._sampler.sample(
            obs=obs_ws,  # all obs so far
            init_state=obs_ws[-1],  # initialize with last obs
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

    def _build_kernel(self):
        """Forward likelihood `p(w_obs | w_true)` for observed data, used as MCMC kernel.

        Args:
            obs_ws (list): list of observed weights
                ntasks * DictList(nchain, ntasks)
            sample_ws (DictList: MCMC sample
                shape nfeats * (nchain,)
            tasks (ndarray): tasks, (ntasks, task_dim)

        Output:
            log_probs (ndarray): log probability of sample_ws (nchain, 1)

        Note:
            * To stabilize MH sampling
              (1) average across tasks (instead of sum)
              (2) average across features costs (instead of sum)

        """

        def likelihood_fn(obs_ws, sample_ws, tasks):
            normal = self._normalizer
            nnorms = len(normal.weights)
            nfeats = sample_ws.num_keys
            nchain = len(sample_ws)
            ntasks = len(tasks)
            try:
                assert obs_ws.shape == (nchain, ntasks)
            except:
                import pdb

                pdb.set_trace()

            ## Pre-empt task computations
            #  shape nfeats * (ntasks,)
            ird_obs_ws = DictList(obs_ws[0])
            #  weights shape nfeats * (ntasks,)
            ird_obs = self.create_particles(
                weights=ird_obs_ws,
                controller=self._sample_controller,
                runner=self._sample_runner,
                save_name="designer_sample",
            )
            ird_obs.get_features_sum(tasks)

            ## Computing Numerator: sample_xxx
            #  weights shape nfeats * (ntasks * nchain,)
            sample_obs = ird_obs.tile(nchain)
            #  shape nfeats * (ntasks * nchain, )
            sample_obs_ws = sample_obs.weights
            #  shape nfeats * (ntasks * nchain, )
            sample_true_ws = sample_ws.repeat(ntasks)
            #  shape (ntasks * nchain, task_dim)
            sample_tasks = onp.tile(tasks, (nchain, 1))
            #  shape (ntasks * nchain,), designer nbatch = ntasks * nchain
            sample_probs = self._designer._kernel(
                sample_true_ws, sample_obs_ws, sample_tasks, sample_obs
            )
            #  shape (ntasks * nchain,) -> (ntasks, nchain)
            sample_probs = sample_probs.reshape((ntasks, nchain))

            ## Computing Denominator: normal_xxx
            #  weights shape nfeats * (ntasks * nnorms,)
            normal_obs = ird_obs.repeat(nnorms)
            #  shape nfeats * (ntasks * nnorms,)
            normal_obs_ws = normal_obs.weights
            #  shape nfeats * (nnorms,)
            normal_true_ws = DictList(normal.weights)
            #  shape nfeats * (ntasks * nnorms,)
            normal_true_ws = normal_true_ws.tile(ntasks)
            #  shape (ntasks * nnorms, task_dim)
            normal_tasks = onp.tile(tasks, (nnorms, 1))
            #  shape (ntasks * nnorms,), Runs (ntasks * nnorms) times
            normal_probs = self._designer._kernel(
                normal_true_ws, normal_obs_ws, normal_tasks, normal_obs
            )

            #  shape (ntasks, nnorms,)
            normal_probs = normal_probs.reshape((ntasks, nnorms))
            normal_prior_probs = self._designer._prior.log_prob(normal_obs.weights)
            normal_prior_probs = normal_prior_probs.reshape((ntasks, nnorms))

            ## Aggregate over tasks
            #  shape (ntasks, nnorms,) -> (ntasks, )
            normal_probs = logsumexp(normal_probs + normal_prior_probs, axis=1)
            # Average across tasks (ntasks, nchain) -> (nchain, )
            log_probs = (sample_probs - normal_probs).mean(axis=0)
            return log_probs

        return likelihood_fn
