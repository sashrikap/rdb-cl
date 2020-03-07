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
import jax.numpy as np
import numpy as onp
from rdb.infer.designer import Designer, DesignerInteractive
from numpyro.handlers import scale, condition, seed
from rdb.infer.particles import Particles
from jax.scipy.special import logsumexp
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
        env_id,
        env_fn,
        controller_fn,
        designer,
        prior_fn,
        ## Weight parameters
        normalized_key,
        num_normalizers=-1,
        proposal_decay=1.0,
        ## Sampling
        task_method="sum",
        sample_method="MH",
        sample_init_args={},
        sample_args={},  # "num_warmups": xx, "num_samples": xx
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
        self._beta = designer.beta
        self._proposal_decay = proposal_decay

        # Normalizer
        self._num_normalizers = num_normalizers
        self._normalizer = None
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
    def interactive_mode(self):
        return self._interactive_mode

    def update_key(self, rng_key):
        """ Update random key """
        ## Sample normaling factor
        self._norm_prior = self._prior_fn("ird_norm")
        self._rng_key, rng_norm = random.split(rng_key, 2)
        sample_prior = seed(self._norm_prior, rng_norm)
        self._normalizer = self.create_particles(
            sample_prior(self._num_normalizers),
            save_name="ird_normalizer",
            runner=self._normal_runner,
            controller=self._normal_controller,
        )

    def _build_controllers(self, controller_fn):
        self._mcmc_controller, self._mcmc_runner = controller_fn(self._env, "IRD MCMC")
        self._sample_controller, self._sample_runner = controller_fn(
            self._env, "IRD Sample"
        )
        self._normal_controller, self._normal_runner = controller_fn(
            self._env, "IRD Normal"
        )
        self._batch_controller, self._batch_runner = controller_fn(
            self._env, "IRD Batch"
        )

    def _build_model(self, observe_ws, tasks):
        """Build IRD PGM model.

        Args:
            observe_ws (DictList): observations seen so far
                shape: nfeats * (ntasks,)
            tasks (ndarray): tasks seen so far
                shape: (ntasks, task_dim)

        Note:
            * To stabilize MH sampling
              (1) average across tasks (instead of sum)
              (2) average across features costs (instead of sum)

        """
        ## Fill in unused feature keys
        feats_keys = self._env.features_keys
        nnorms = len(self._normalizer.weights)
        nnorms_1 = nnorms + 1
        dnnorms = len(self._designer._normalizer.weights)
        nfeats = len(self._env.features_keys)
        ntasks = len(tasks)

        ## ==========================================
        ## ======= Pre-empt heavy optimiations ======
        #  shape nfeats * (ntasks,)
        obs_ps = self.create_particles(
            weights=observe_ws,
            controller=self._sample_controller,
            runner=self._sample_runner,
        )
        #  shape (nfeats, ntasks)
        obs_ws = observe_ws.prepare(feats_keys).numpy_array()
        #  shape (nfeats, 1, ntasks)
        obs_feats_sum = (
            obs_ps.get_features_sum(tasks)
            .prepare(feats_keys)[np.diag_indices(ntasks)]
            .expand_dims(0)
            .numpy_array()
        )
        #  shape (ntasks, 1, T, udim)
        obs_actions = obs_ps.get_actions(tasks)[np.diag_indices(ntasks)][:, None]
        # #  shape (ntasks, nnorms, T, udim)
        # obs_actions_norm = np.repeat(obs_actions, nnorms, axis=0)
        # #  shape (ntasks, dnnorms, T, udim)
        # obs_actions_dnorm = np.repeat(obs_actions, dnnorms, axis=0)

        normal_ws = self._normalizer.weights
        normal_ws = normal_ws.prepare(feats_keys).numpy_array()

        ## =================== Prepare IRD Normalizer ==================
        # self._normalizer.compute_tasks(tasks, us0=obs_actions_norm)
        self._normalizer.compute_tasks(tasks, max_batch=50)
        #  shape (nfeats, ntasks, nnorms)
        ird_normal_fsum = self._normalizer.get_features_sum(tasks)
        ird_normal_fsum = ird_normal_fsum.prepare(feats_keys).numpy_array()
        #  shape (nnorms, nfeats, ntasks)
        ird_normal_fsum = ird_normal_fsum.swapaxes(0, 2).swapaxes(1, 2)
        #  shape (nnorms, nfeats, 1, ntasks)
        ird_normal_fsum = ird_normal_fsum[:, :, None]

        ## ================= Prepare Designer Normalizer ================
        # self._designer._normalizer.compute_tasks(tasks, us0=obs_actions_dnorm)
        self._designer._normalizer.compute_tasks(tasks, max_batch=50)
        #  shape (nfeats, ntasks, d_nnorms)
        designer_normal_fsum = self._designer.normalizer.get_features_sum(tasks)
        designer_normal_fsum = designer_normal_fsum.prepare(feats_keys).numpy_array()
        #  shape (nfeats, 1, ntasks, d_nnorms)
        designer_normal_fsum = designer_normal_fsum[:, None]
        #  shape (1, ntasks, task_dim)
        obs_tasks = tasks[None, :]

        assert obs_ws.shape == (nfeats, ntasks)
        assert normal_ws.shape == (nfeats, nnorms)
        assert obs_feats_sum.shape == (nfeats, 1, ntasks)
        assert ird_normal_fsum.shape == (nnorms, nfeats, 1, ntasks)

        ## Designer kernels
        designer_ll = partial(
            self._designer.likelihood_ird,
            true_feats_sum=obs_feats_sum,
            sample_ws=obs_ws,
            tasks=obs_tasks,
            sample_feats_sum=obs_feats_sum,
            normal_feats_sum=designer_normal_fsum,
        )
        designer_vll = jax.vmap(designer_ll)

        ## Operation over tasks
        task_method = None
        if self._task_method == "sum":
            task_method = np.sum
        elif self._task_method == "mean":
            task_method = np.mean
        else:
            raise NotImplementedError

        def _model():
            #  shape (nfeats, 1)
            true_ws = self._prior(1).prepare(feats_keys)

            # ## ======= Not jit-able optimization: requires scipy/jax optimizer ======
            # true_ps = self.create_particles(
            #     true_ws, controller=self._sample_controller, runner=self._sample_runner
            # )
            # true_ps.compute_tasks(tasks, us0=obs_actions, jax=False)
            # #  shape (nfeats, ntasks, 1)
            # true_feats_sum = true_ps.get_features_sum(tasks).prepare(feats_keys)
            # true_feats_sum = true_feats_sum.numpy_array()
            # #  shape (nfeats, 1, ntasks)
            # true_feats_sum = true_feats_sum.swapaxes(1, 2)

            # ## ======= Jit-able cheap alternative ======
            true_feats_sum = obs_feats_sum

            #  shape (nnorms_1, nfeats, 1, ntasks)
            ird_normal_fsum_1 = np.concatenate(
                [true_feats_sum[None, :], ird_normal_fsum], axis=0
            )
            true_ws = true_ws.numpy_array()
            log_prob = _likelihood(true_ws, true_feats_sum, ird_normal_fsum_1)
            numpyro.factor("ird_log_probs", log_prob)

        #  shape (nnorms, nfeats, ntasks)
        normal_ws_v = np.repeat(normal_ws.swapaxes(0, 1)[:, :, None], ntasks, axis=2)
        #  shape (nnorms, ntasks)
        ird_normal_probs = designer_vll(normal_ws_v)

        @jax.jit
        def _likelihood(ird_true_ws, ird_true_feats_sum, ird_normal_feats_sum):
            """
            Forward likelihood `p(w_obs | w_true)` for observed data, used as MCMC kernel.

            Args:
                ird_true_ws (ndarray): sampled weights
                    shape: (nfeats, 1)
                ird_true_feats_sum (ndarray): features of sampled weights on tasks
                    shape: (nfeats, 1, ntasks)
                ird_normal_feats_sum (ndarray): features of normal weights on tasks
                    shape: (nnorms_1, nfeats, 1, ntasks)

            Output:
                log_probs (ndarray): log probability of ird_true_ws (1, )

            TODO:
                * Dynamically changing feature counts

            """
            ## ===============================================
            ## ======= Computing Numerator: sample_xxx =======
            #  shape (nfeats, ntasks)
            ird_true_ws_n = np.repeat(ird_true_ws, ntasks, axis=1)
            #  shape (ntasks,), designer nbatch = ntasks
            # ird_true_probs = designer_ll(ird_true_ws_n, ird_true_feats_sum)
            ird_true_probs = designer_ll(ird_true_ws_n)
            assert ird_true_probs.shape == (ntasks,)

            # ## =================================================
            # ## ======= Computing Denominator: normal_xxx =======
            # #  shape (nnorms_1, nfeats)
            # normal_ws_1 = np.concatenate([ird_true_ws, normal_ws], 1).swapaxes(0, 1)
            # #  shape (nnorms_1, nfeats, ntasks)
            # normal_ws_1 = np.repeat(normal_ws_1[:, :, None], ntasks, axis=2)
            # #  shape (nnorms_1, ntasks)
            # # ird_normal_probs = designer_vll(normal_ws_1, ird_normal_feats_sum)
            # ird_normal_probs = designer_vll(normal_ws_1)

            ## =================================================
            ## ======= Computing Denominator: normal_xxx =======
            #  shape (1, nfeats, ntasks)
            ird_true_ws_v = np.repeat(
                ird_true_ws.swapaxes(0, 1)[:, :, None], ntasks, axis=2
            )
            #  shape (1, ntasks)
            ird_true_probs = designer_vll(ird_true_ws_v)
            assert ird_true_probs.shape == (1, ntasks)
            #  shape (nnorms_1, ntasks)
            ird_normal_probs_ = np.concatenate(
                [ird_true_probs, ird_normal_probs], axis=0
            )

            #  shape (nnorms_1, ntasks) -> (ntasks, )
            ird_normal_probs_ = logsumexp(ird_normal_probs_, axis=0) - np.log(nnorms_1)
            assert ird_normal_probs_.shape == (ntasks,)

            ## ==================================================
            ## ================= Aggregate tasks ================
            log_prob = task_method(ird_true_probs - ird_normal_probs_)
            return log_prob

        # import pdb; pdb.set_trace()
        # # d_true_ps = self.create_particles(
        # #     self._designer.truth.weights, controller=self._sample_controller, runner=self._sample_runner
        # # )
        # # d_true_ps.compute_tasks(tasks, us0=obs_actions, jax=False)
        # # #  shape (nfeats, ntasks, 1)
        # # true_feats_sum = d_true_ps.get_features_sum(tasks).prepare(feats_keys)
        # # true_feats_sum = true_feats_sum.numpy_array()
        # # #  shape (nfeats, 1, ntasks)
        # # true_feats_sum = true_feats_sum.swapaxes(1, 2)

        # true_feats_sum = obs_feats_sum
        # ird_normal_fsum_1 = np.concatenate([true_feats_sum[None, :], ird_normal_fsum], axis=0)
        # d_joint_ws = DictList(OrderedDict([('control', 2), ('dist_fences', 5.35), ('dist_lanes', 2.5), ('dist_objects', 4.25), ('speed', 5), ('speed_over', 80), ('dist_cars', 2.0)]), expand_dims=True).normalize_by_key("dist_cars")
        # #d_true_ws = self._designer.truth.weights.prepare(feats_keys).numpy_array()
        # d_joint_ws = d_joint_ws.prepare(feats_keys).numpy_array()
        # log_prob_joint = _likelihood(d_joint_ws, true_feats_sum, ird_normal_fsum_1)

        # b_map_ws = DictList(OrderedDict([('control', 2.0773953897444115), ('dist_fences', 161.915494922572), ('dist_lanes', 7.173069941020064), ('dist_objects', 16.272937241410762), ('speed', 0.10904018090865127), ('speed_over', 129.47986491349732), ('dist_cars', 1.0)]), expand_dims=True).normalize_by_key("dist_cars")
        # b_map_ws = b_map_ws.prepare(feats_keys).numpy_array()
        # log_prob_map = _likelihood(b_map_ws, true_feats_sum, ird_normal_fsum_1)

        # b_map_ps = self.create_particles(b_map_ws, controller=self._sample_controller, runner=self._sample_runner)
        # b_map_ps.compute_tasks(tasks, us0=obs_actions, jax=False)
        # #  shape (nfeats, ntasks, 1)
        # b_map_feats_sum = b_map_ps.get_features_sum(tasks).prepare(feats_keys)
        # b_map_feats_sum = b_map_feats_sum.numpy_array()
        # #  shape (nfeats, 1, ntasks)
        # b_map_feats_sum = b_map_feats_sum.swapaxes(1, 2)

        # b_map_ws = b_map_ws.prepare(feats_keys).numpy_array()
        # idx=0
        # _likelihood(b_map_ws, true_feats_sum, ird_normal_fsum_1)
        # b_map_ws = DictList({'control': array([0.0965125]), 'dist_cars': array([1.]), 'dist_fences': array([295.96623983]), 'dist_lanes': array([0.10440017]), 'dist_objects': array([0.3627585]), 'speed': array([0.36979608]), 'speed_over': array([11.51023696])})
        # b_map_ws = b_map_ws.prepare(feats_keys).numpy_array()
        # idx=0
        # _likelihood(b_map_ws, true_feats_sum, ird_normal_fsum_1)
        return _model

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

        """
        feats_keys = self._env.features_keys
        tasks = onp.array(tasks)
        ntasks = len(tasks)
        assert len(tasks) > 0, "Need >=1 tasks"
        assert len(tasks) == len(obs), "Tasks and observations mismatch"
        assert self._normalizer is not None

        ## ==============================================================
        ## =================== Prepare for Sampling =====================
        #  shape nfeats * (ntasks,)
        observe_ws = DictList(
            [ob.weights.prepare(feats_keys) for ob in obs], jax=True
        ).squeeze(1)
        self._update_normalizer(obs)

        ## ==============================================================
        ## ====================== MCMC Sampling =========================
        init_args = copy.deepcopy(self._sample_init_args)
        samp_args = copy.deepcopy(self._sample_args)
        decay = self._proposal_decay ** (ntasks - 1)
        init_args["proposal_var"] = init_args["proposal_var"] * decay

        ## ================= Build Prior Function with necessary keys ================
        self._prior = self._prior_fn("ird")
        for ob in obs:
            for key in ob.weights.keys():
                if key != self._normalized_key:
                    self._prior.add_feature(key)

        num_chains = self._sample_args["num_chains"]
        num_keys = len(self._prior.feature_keys)
        init_params = DictList(
            dict(zip(self._prior.feature_keys, np.zeros((num_keys, 1))))
        )
        if num_chains > 1:
            #  shape nfeats * (nchains, 1)
            init_params = init_params.expand_dims(0).repeat(num_chains, axis=0)
        del init_params[self._normalized_key]

        ird_model = self._build_model(observe_ws, tasks)
        sampler = get_ird_sampler(self._sample_method, ird_model, init_args, samp_args)
        # sampler = get_designer_sampler(self._sample_method, ird_model, init_args, samp_args)

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
        sample_ws[self._normalized_key] = np.zeros(sample_ws.shape)
        sample_info = sampler.get_extra_fields(group_by_chain=True)
        sample_rates = sample_info["mean_accept_prob"][:, -1]
        num_samples = sample_ws.shape[1]
        visualize_chains(
            chains=sample_ws,
            rates=sample_rates,
            fig_dir=f"{self._save_dir}/mcmc",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples}",
            **self._weight_params,
        )
        samples = self.create_particles(
            sample_ws[0].exp(),
            save_name=save_name,
            runner=self._sample_runner,
            controller=self._sample_controller,
        )
        samples.visualize(true_w=self.designer.true_w, obs_w=observe_ws[-1])
        return samples

    def _update_normalizer(self, obs):
        """
        If user add feature during designing, update normalizer with new feature keys.
        """
        rebuild_normalizer = False
        all_keys = []
        for ob in obs:
            for key in ob.weights.keys():
                all_keys.append(key)
                if (
                    key not in self._norm_prior.feature_keys
                    and key != self._normalized_key
                ):
                    rebuild_normalizer = True
                    self._norm_prior.add_feature(key)
        all_keys = set(all_keys)
        self._norm_prior = self._prior_fn(name="ird_norm", feature_keys=all_keys)
        if rebuild_normalizer:
            self._rng_key, rng_norm = random.split(self._rng_key)
            sample_prior = seed(self._norm_prior, rng_norm)
            self._normalizer = self.create_particles(
                sample_prior(self._num_normalizers),
                save_name="ird_normalizer",
                runner=self._normal_runner,
                controller=self._normal_controller,
            )

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
