"""Inverse Reward Design Module for Optimal Control.

Includes:
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
import jax.numpy as jnp
import numpy as onp
from rdb.infer.designer import Designer, DesignerInteractive
from rdb.infer.particles import Particles
from jax.scipy.special import logsumexp
from rdb.exps.utils import Profiler
from tqdm.auto import tqdm, trange
from rdb.infer.utils import *
from rdb.exps.utils import *
from os.path import join
from jax import random
from time import time

pp = pprint.PrettyPrinter(indent=4)


class IRDOptimalControl(object):
    """Inverse Reward Design for Optimal Control.

    Given environment `env`, `planner`, user input w
    infer p(w* | w).

    Notes:
        * To reduce JIT recompile time, uses separate controller for(1) normalizers,
        (2) batch kernel, (3) samples and (4) designer

    Args:
        controller_fn (fn): returns controller and runner
            controller: `actions = controller(state, w)`
            runner: `traj, cost, info = runner(state, actions)`
        beta (float): temperature param
            `p ~ exp(beta * reward)`
        prior (Prior): sampling prior
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
        env_id,
        env_fn,
        controller_fn,
        beta,
        designer,
        prior_fn,
        ## Weight parameters
        normalized_key,
        proposal_decay=1.0,
        max_val=15.0,
        ## Sampling
        task_method="sum",
        sample_method="MH",
        sample_init_args={},
        sample_args={},  # "num_warmups": xx, "num_samples": xx
        mcmc_normalize=None,
        ## Parameter for histogram
        weight_params={},
        interactive_mode=False,
        interactive_name="Default",
        ## Saving options
        save_root="data",
        exp_name="active_ird_exp1",
    ):
        self._rng_key = None
        self._rng_name = None
        # Environment settings
        self._env_id = env_id
        self._env_fn = env_fn
        self._env = env_fn()
        self._build_controllers(controller_fn)
        self._interactive_mode = interactive_mode

        # Rationality
        self._designer = designer
        self._beta = beta
        self._proposal_decay = proposal_decay
        self._max_val = max_val

        # Normalizer
        self._normalized_key = normalized_key
        self._weight_params = weight_params

        # Sampling functions
        self._prior_fn = prior_fn
        self._prior = None
        self._model = None
        self._task_method = task_method
        self._sampler = None
        self._sample_args = sample_args
        self._sample_method = sample_method
        self._sample_init_args = sample_init_args
        self._mcmc_normalize = mcmc_normalize

        ## Caching normalizers, samples and features
        self._user_actions = {}
        self._user_feats = {}

        ## Saving
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"

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
    def rng_name(self):
        return self._rng_name

    @rng_name.setter
    def rng_name(self, name):
        self._rng_name = name

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def interactive_mode(self):
        return self._interactive_mode

    def update_key(self, rng_key):
        """ Update random key """
        ## Sample normaling factor
        self._rng_key, rng_norm, rng_designer = random.split(rng_key, 3)
        self._designer.update_key(rng_designer)

    def _build_controllers(self, controller_fn):
        self._mcmc_controller, self._mcmc_runner = controller_fn(self._env, "IRD MCMC")
        self._sample_controller, self._sample_runner = controller_fn(
            self._env, "IRD Sample"
        )
        self._obs_controller, self._obs_runner = controller_fn(self._env, "IRD Obs")

    def update_prior(self, keys):
        for key in keys:
            if key != self._normalized_key:
                self._prior.add_feature(key)

    def _build_model(self, obs, tasks, universal_model=False, expensive=False):
        """Build IRD PGM model.

        Args:
            obs (list): list of observations seen so far
            tasks (ndarray): tasks seen so far
                shape: (ntasks, task_dim)
            universal_model (jax.model): universal planning network
            expensive (bool): include expensive inner-opt operation during sampling

        Note:
            * To stabilize MH sampling
              (1) average across tasks (instead of sum)
              (2) average across features costs (instead of sum)

        """
        ## Fill in unused feature keys
        feats_keys = self._env.features_keys
        dnnorms = len(self._designer.normalizer.weights)
        nfeats = len(self._env.features_keys)
        ntasks = len(tasks)

        ## ==============================================================
        ## =========== Build Prior Function with necessary keys ==========
        self._prior = self._prior_fn("ird")
        for ob in obs:
            self.update_prior(ob.weights.keys())
            self._designer.update_prior(ob.weights.keys())

        if universal_model is None:
            ## ==========================================
            ## ======= Pre-empt heavy optimiations ======
            #  shape nfeats * (ntasks,)
            obs_ws = DictList(
                [ob.weights.prepare(feats_keys) for ob in obs], jax=True
            ).squeeze(1)
            obs_ps = self.create_particles(
                weights=obs_ws, controller=self._obs_controller, runner=self._obs_runner
            )
            #  shape (nfeats, ntasks)
            obs_ws = obs_ws.prepare(feats_keys).normalize_across_keys().numpy_array()
            # obs_ws = obs_ws.prepare(feats_keys).numpy_array()
            #  shape (nfeats, 1, ntasks)
            obs_feats_dict = obs_ps.get_features_sum(tasks)
            obs_feats_sum = (
                obs_ps.get_features_sum(tasks)
                .prepare(feats_keys)[jnp.diag_indices(ntasks)]
                .expand_dims(0)
                .numpy_array()
            )
            if self._mcmc_normalize == "hessian":
                _, obs_hnorm = obs_ps.get_hessians(tasks)

            #  shape (ntasks, 1, T, udim)
            # obs_actions = obs_ps.get_actions(tasks)[jnp.diag_indices(ntasks)][:, None]
            # #  shape (ntasks, dnnorms, T, udim)
            # obs_actions_dnorm = jnp.repeat(obs_actions, dnnorms, axis=0)

            ## ================= Prepare Designer Normalizer ================
            # self._designer.normalizer.compute_tasks(tasks, us0=obs_actions_dnorm)
            print(f"Computing designer normalizers: {self._designer.num_normalizers}")
            t1 = time()
            self._designer.normalizer.compute_tasks(tasks, max_batch=500)
            print(f"Computing designer normalizers time {time() - t1}")
            #  shape (nfeats, ntasks, d_nnorms)
            designer_normal_fsum = self._designer.normalizer.get_features_sum(tasks)
            designer_normal_fsum = designer_normal_fsum.prepare(
                feats_keys
            ).numpy_array()
            #  shape (nfeats, 1, ntasks, d_nnorms)
            designer_normal_fsum = designer_normal_fsum[:, None]
            #  shape (nfeats, d_nnorms)
            designer_normal_ws = self._designer.normalizer.weights.prepare(
                feats_keys
            ).numpy_array()

            #  shape (1, ntasks, task_dim)
            obs_tasks = tasks[None, :]
        else:
            ## ==========================================
            ## ======= Pre-empt heavy optimiations ======
            #  shape nfeats * (ntasks,)
            obs_ws = DictList(
                [ob.weights.prepare(feats_keys) for ob in obs], jax=True
            ).squeeze(1)
            # obs_ps = self.create_particles(
            #     weights=obs_ws,
            #     controller=self._sample_controller,
            #     runner=self._sample_runner,
            # )
            #  shape (nfeats, ntasks)
            obs_ws = obs_ws.prepare(feats_keys).normalize_across_keys().numpy_array()
            #  shape (1, ws_dim)
            obs_ws_arr = jnp.array(list(ob.weights.values())).T
            #  shape (nfeats, 1, ntasks)
            obs_feats_sum = universal_model(obs_ws_arr).T[:, None, :]
            # obs_feats_sum = (
            #     obs_ps.get_features_sum(tasks)
            #     .prepare(feats_keys)[jnp.diag_indices(ntasks)]
            #     .expand_dims(0)
            #     .numpy_array()
            # )
            #  shape nfeats * (1, ntasks)
            obs_feats_dict = DictList(ob.weights)
            obs_feats_dict.from_array(obs_feats_sum)

            print(f"Computing designer normalizers: {self._designer.num_normalizers}")
            t1 = time()
            #  shape (n_norm, ws_dim)
            norm_ws_arr = jnp.array(list(self._designer.normalizer.weights.values())).T
            ## TODO: only 1 task supported
            norm_feats_sum = universal_model(norm_ws_arr).T[:, None, :]
            print(f"Computing designer normalizers time {time() - t1}")

            #  shape (nfeats, ntasks, d_nnorms)
            designer_normal_fsum = norm_feats_sum
            #  shape (nfeats, 1, ntasks, d_nnorms)
            designer_normal_fsum = designer_normal_fsum[:, None]
            #  shape (nfeats, d_nnorms)
            designer_normal_ws = self._designer.normalizer.weights.prepare(
                feats_keys
            ).numpy_array()

            #  shape (1, ntasks, task_dim)
            obs_tasks = tasks[None, :]

        assert obs_ws.shape == (nfeats, ntasks)
        assert obs_feats_sum.shape == (nfeats, 1, ntasks)

        ## Designer kernels
        designer_ll = partial(
            self._designer.likelihood_ird,
            sample_ws=obs_ws,
            normal_feats_sum=designer_normal_fsum,
            tasks=obs_tasks,
            beta=self._designer.beta,
        )

        ## Operation over tasks
        task_method = None
        if self._task_method == "sum":
            task_method = jnp.sum
        elif self._task_method == "mean":
            task_method = jnp.mean
        else:
            raise NotImplementedError

        if universal_model is None:

            def _model():
                #  shape (nfeats, 1)
                nonlocal tasks
                nonlocal ntasks
                nonlocal obs_feats_sum
                nonlocal designer_ll
                nonlocal designer_normal_ws  # shape (nfeats, d_nnorms)
                nonlocal designer_normal_fsum
                true_ws = self._prior(1)  # nfeats * (1,)
                log_prior = jnp.log(true_ws.numpy_array())
                infeasible = jnp.logical_or(
                    jnp.any(log_prior < -1 * self._max_val),
                    jnp.any(log_prior > self._max_val),
                )
                true_ws = true_ws.prepare(feats_keys)

                if expensive:
                    ## ======= Not jit-able optimization: requires scipy/jax optimizer ======
                    true_ps = self.create_particles(
                        true_ws,
                        controller=self._sample_controller,
                        runner=self._sample_runner,
                    )
                    true_ps.compute_tasks(tasks, jax=True, max_batch=ntasks)
                    #  shape (nfeats, ntasks, 1)
                    true_feats_sum = true_ps.get_features_sum(tasks).prepare(feats_keys)
                    #  shape (nfeats, 1, ntasks)
                    true_feats_sum = true_feats_sum.numpy_array().swapaxes(1, 2)
                else:
                    ## ======= Jit-able cheap alternative ======
                    true_feats_sum = obs_feats_sum

                ## Weight offset, only used when normalize mode is `offset`
                true_offset = jnp.zeros(ntasks)

                if self._mcmc_normalize == "hessian":
                    _, true_hnorm = obs_ps.get_hessians(tasks, true_ws)
                    true_hnorm /= obs_hnorm  # normalize by observation hessian sum
                    true_ws = true_ws.numpy_array() / true_hnorm
                elif self._mcmc_normalize == "magnitude":
                    true_ws = true_ws.normalize_across_keys().numpy_array()
                elif self._mcmc_normalize == "offset":
                    # shape (1,)
                    # true_ws = true_ws.normalize_across_keys().numpy_array()
                    true_ws = true_ws.numpy_array()
                    true_offset = -(true_ws * obs_feats_sum[:, 0]).sum(axis=0)
                elif self._mcmc_normalize is None:
                    true_ws = true_ws.numpy_array()
                else:
                    raise NotImplementedError
                log_prob = jnp.where(
                    infeasible, -1e10, _likelihood(true_ws, true_feats_sum, true_offset)
                )

                numpyro.factor("ird_log_probs", log_prob)

        else:

            def _model():
                #  shape (nfeats, 1)
                nonlocal tasks
                nonlocal ntasks
                nonlocal obs_feats_sum
                nonlocal designer_ll
                nonlocal designer_normal_ws  # shape (nfeats, d_nnorms)
                nonlocal designer_normal_fsum
                true_ws = self._prior(1)
                log_prior = jnp.log(true_ws.numpy_array())
                infeasible = jnp.logical_or(
                    jnp.any(log_prior < -1 * self._max_val),
                    jnp.any(log_prior > self._max_val),
                )

                sample_input = true_ws.numpy_array().T

                # ## ======= Jit-able cheap alternative ======
                true_feats_sum = universal_model(sample_input)

                ## Weight offset, only used when normalize mode is `offset`
                true_offset = jnp.zeros(ntasks)
                true_ws = true_ws.prepare(feats_keys).numpy_array()
                log_prob = jnp.where(
                    infeasible, -1e10, _likelihood(true_ws, true_feats_sum, true_offset)
                )
                numpyro.factor("ird_log_probs", log_prob)

        @jax.jit
        def _likelihood(ird_true_ws, ird_true_feats_sum, true_offset):
            """
            Forward likelihood `p(w_obs | w_true)` for observed data, used as MCMC kernel.

            Args:
                ird_true_ws (ndarray): sampled weights
                    shape: (nfeats, 1)
                ird_true_feats_sum (ndarray): features of sampled weights on tasks
                    shape: (nfeats, 1, ntasks)
                true_offset (ndarray): offset for sampled weights
                    shape: (1,)

            Output:
                log_probs (ndarray): log probability of ird_true_ws (1, )

            TODO:
                * Dynamically changing feature counts

            """
            nonlocal obs_feats_sum
            nonlocal designer_normal_fsum
            ## ===============================================
            ## ======= Computing Numerator: sample_xxx =======
            #  shape (nfeats, ntasks)
            ird_true_ws_n = jnp.repeat(ird_true_ws, ntasks, axis=1)
            #  shape (ntasks,), designer nbatch = ntasks

            ird_true_probs = self._beta * designer_ll(
                true_ws=ird_true_ws_n,
                true_offset=true_offset,
                true_feats_sum=ird_true_feats_sum,
            )
            # ird_true_probs = designer_ll(ird_true_ws_n, ird_true_feats_sum)
            assert ird_true_probs.shape == (ntasks,)
            ## ==================================================
            ## ================= Aggregate tasks ================
            # log_prob = task_method(ird_true_probs - ird_normal_probs_)
            log_prob = task_method(ird_true_probs)
            return log_prob

        return _model

    def load_sample(self, save_name):
        samples = self.create_particles(
            [],
            save_name=save_name,
            runner=self._sample_runner,
            controller=self._sample_controller,
        )
        samples.load()
        return samples

    def sample(
        self, tasks, obs, save_name, verbose=True, universal_model=None, expensive=False
    ):
        """Sample b(w) for true weights given obs.weights.

        Args:
            tasks (list): list of tasks so far
            obs (list): list of observation particles, each for 1 task

        Note:
            * To approximate IRD doubly-intractible denominator:
                `sample`: use random samples for normalizer;
                `max_norm`: use max trajectory as normalizer;
                `hybrid`: use random samples mixed with max trajectory

        """
        feats_keys = self._env.features_keys
        tasks = onp.array(tasks)
        ntasks = len(tasks)
        assert len(tasks) > 0, "Need >=1 tasks"
        assert len(tasks) == len(obs), "Tasks and observations mismatch"

        ## ==============================================================
        ## ====================== MCMC Sampling =========================
        init_args = copy.deepcopy(self._sample_init_args)
        samp_args = copy.deepcopy(self._sample_args)
        decay = self._proposal_decay ** (ntasks - 1)
        init_args["proposal_var"] = init_args["proposal_var"] * decay
        ird_model = self._build_model(
            obs, tasks, universal_model=universal_model, expensive=expensive
        )

        num_chains = self._sample_args["num_chains"]
        num_keys = len(self._prior.feature_keys)
        init_params = DictList(
            dict(zip(self._prior.feature_keys, jnp.zeros((num_keys, 1))))
        )
        # init_params = obs[-1].weights.log()
        if num_chains > 1:
            #  shape nfeats * (nchains, 1)
            init_params = init_params.expand_dims(0).repeat(num_chains, axis=0)
        if self._normalized_key in init_params:
            del init_params[self._normalized_key]

        if not expensive:
            sampler = get_ird_sampler(
                self._sample_method, ird_model, init_args, samp_args
            )
        else:
            # Include expensive inner-opt operation during sampling
            sampler = get_designer_sampler(
                self._sample_method, ird_model, init_args, samp_args
            )

        self._rng_key, rng_sampler = random.split(self._rng_key, 2)
        print(f"Sampling IRD (obs={len(obs)}): {save_name}, chains={num_chains}")
        t1 = time()
        sampler.run(
            rng_sampler,
            init_params=dict(init_params),
            extra_fields=["mean_accept_prob"],
        )

        ## ==============================================================
        ## ====================== Analyze Samples =======================
        #  shape nfeats * (nchains, nsample, 1) -> (nchains, nsample)
        sample_ws = sampler.get_samples(group_by_chain=True)
        sample_ws = DictList(sample_ws, jax=True).squeeze(axis=2)
        print(f"Sample time {time() - t1}")

        assert self._normalized_key not in sample_ws.keys()
        if self._normalized_key in self._prior.feature_keys:
            sample_ws[self._normalized_key] = jnp.zeros(sample_ws.shape)
        sample_info = sampler.get_extra_fields(group_by_chain=True)
        sample_rates = sample_info["mean_accept_prob"][:, -1]
        num_samples = sample_ws.shape[1]
        # Visualize histogram per feature
        visualize_mcmc_feature(
            chains=sample_ws,
            rates=sample_rates,
            fig_dir=f"{self._save_dir}/mcmc",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples}_feature",
            **self._weight_params,
        )
        # Visualize distribution per feature pairs
        visualize_mcmc_pairs(
            chains=sample_ws,
            fig_dir=f"{self._save_dir}/mcmc/pairs",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples}_pairs",
            **self._weight_params,
        )
        samples = self.create_particles(
            sample_ws[0].exp(),
            save_name=save_name,
            runner=self._sample_runner,
            controller=self._sample_controller,
        )
        last_ws = obs[-1].weights.prepare(feats_keys)[0]
        samples.visualize(true_w=self.designer.true_w, obs_w=last_ws)
        return samples

    def create_particles(self, weights, runner, controller, save_name=""):
        if weights is not None:
            weights = DictList(weights)
        self._rng_key, rng_particle = random.split(self._rng_key, 2)
        return Particles(
            rng_name=self._rng_name,
            rng_key=rng_particle,
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
