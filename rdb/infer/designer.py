import os
import jax
import copy
import json
import time
import pprint
import numpy as onp
import jax.numpy as jnp
import numpyro, itertools
from jax import random
from os.path import join
from rdb.exps.utils import *
from functools import partial
from rdb.infer.utils import *
from numpyro.handlers import seed
from tqdm.auto import tqdm, trange
from rdb.infer.dictlist import DictList
from jax.scipy.special import logsumexp
from rdb.infer.particles import Particles


class Designer(object):
    """Simulated designer module.

    Given true weights, sample MAP design weights.

    Note:
        * Currently assumes same beta, proposal as IRD module, although
          doesn't have to be.

    """

    def __init__(
        self,
        env_fn,
        controller_fn,
        beta,
        prior_fn,
        ## Weight parameters
        normalized_key,
        num_normalizers,
        proposal_decay=1.0,
        max_val=15.0,
        ## Sampling
        prior_keys=None,
        use_true_w=False,
        design_mode="independent",
        select_mode="mean",
        task_method="sum",
        sample_method="MH",
        sample_init_args={},
        sample_args={},
        ## Parameter for histogram
        weight_params={},
        ## Saving options
        save_root="",
        exp_name="",
        prior_tasks=[],
    ):
        self._rng_key = None
        self._rng_name = None
        # Environment settings
        self._env_fn = env_fn
        self._env = env_fn()
        self._build_controllers(controller_fn)
        self._true_w = None
        self._truth = None
        self._beta = beta
        self._normalized_key = normalized_key
        self._weight_params = weight_params

        ## Normalizer
        self._num_normalizers = num_normalizers
        self._normalizer = None
        self._norm_prior = None
        self._proposal_decay = proposal_decay
        self._max_val = max_val

        ## Designer model
        self._use_true_w = use_true_w
        assert design_mode in {"independent", "joint"}
        self._design_mode = design_mode
        assert select_mode in {"mean", "map"}
        self._select_mode = select_mode

        ## Sampling Prior and kernel
        self._prior_fn = prior_fn
        self._prior_keys = prior_keys
        self._prior = None
        self._model = None
        self._task_method = task_method
        self._likelihood, self._likelihood_ird = self._build_likelihood()
        self._sample_args = sample_args
        self._sample_method = sample_method
        self._sample_init_args = sample_init_args

        ## Saving
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"

        ## Designer Prior tasks
        self._prior_tasks = []

    @property
    def use_true_w(self):
        return self._use_true_w

    @property
    def design_mode(self):
        return self._design_mode

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def likelihood_ird(self):
        return self._likelihood_ird

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def num_normalizers(self):
        return self._num_normalizers

    @property
    def rng_name(self):
        return self._rng_name

    @rng_name.setter
    def rng_name(self, name):
        self._rng_name = name
        self._normalizer.rng_name = name

    def _build_controllers(self, controller_fn):
        self._controller, self._runner = controller_fn(self._env, "Designer Core")
        self._norm_controller, self._norm_runner = controller_fn(
            self._env, "Designer Norm"
        )
        self._true_controller, self._true_runner = controller_fn(
            self._env, "Designer True"
        )
        self._one_controller, self._one_runner = controller_fn(
            self._env, "Designer One"
        )
        self._sample_controller, self._sample_runner = controller_fn(
            self._env, "Designer Sample"
        )

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

    @true_w.setter
    def true_w(self, w):
        print("Designer truth updated")
        w_dict = DictList([w], jax=False).normalize_by_key(self._normalized_key)
        self._true_w = w_dict[0]
        self._truth = self.create_particles(
            weights=w_dict,
            controller=self._true_controller,
            runner=self._true_runner,
            save_name="designer_truth",
        )

    @property
    def truth(self):
        return self._truth

    @property
    def env(self):
        return self._env

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    def update_key(self, rng_key):
        self._rng_key, rng_truth, rng_norm = random.split(rng_key, 3)
        ## Sample normaling factor
        self._norm_prior = seed(self._prior_fn("designer_norm"), rng_norm)
        self._normalizer = self.create_particles(
            self._norm_prior(self._num_normalizers),
            save_name="designer_normalizer",
            runner=self._norm_runner,
            controller=self._norm_controller,
        )
        ## Build likelihood and model
        self._prior = self._prior_fn("designer")
        if self._truth is not None:
            self._truth.update_key(rng_truth)

    def update_prior(self, keys):
        update_normalizer = False
        for key in keys:
            if key != self._normalized_key and key not in self._prior.feature_keys:
                self._prior.add_feature(key)
                update_normalizer = True
                print(f"Designer prior updated: {key}")
        if update_normalizer:
            keys = self._prior.feature_keys
            self._rng_key, rng_norm = random.split(self._rng_key, 2)
            self._norm_prior = seed(self._prior_fn("designer_norm", keys), rng_norm)
            self._normalizer = self.create_particles(
                self._norm_prior(self._num_normalizers),
                save_name="designer_normalizer",
                runner=self._norm_runner,
                controller=self._norm_controller,
            )

    def simulate(
        self, tasks, save_name, tqdm_position=0, universal_model=None, mode="importance"
    ):
        if mode == "mcmc":
            return self.simulate_mcmc(tasks, save_name, tqdm_position, universal_model)
        elif mode == "importance":
            return self.simulate_impt(tasks, save_name, tqdm_position, universal_model)
        else:
            raise NotImplementedError

    def simulate_impt(self, tasks, save_name, tqdm_position=0, universal_model=None):
        """Sample weights (#normalizer) and compute importance weights.
        """
        feats_keys = self._env.features_keys

        nfeats = len(feats_keys)
        ntasks = len(tasks)
        nnorms = len(self._normalizer.weights)
        #  shape nfeats * (ntasks, nnorm)
        norm_feats = self._normalizer.get_features_sum(tasks).prepare(feats_keys)

        task_method = None
        if self._task_method == "sum":
            task_method = partial(jnp.sum, axis=0)
        elif self._task_method == "mean":
            task_method = partial(jnp.mean, axis=0)
        else:
            raise NotImplementedError

        #  shape (ntasks, nnorm)
        true_ws = DictList([self._true_w]).prepare(feats_keys)
        true_arr = true_ws.expand_dims(-1).numpy_array()
        #  shape (ntasks, nnorm)
        imp_weights = -1 * (true_arr * norm_feats.numpy_array()).sum(axis=0)

        #  shape (nnorm, )
        imp_weights = task_method(imp_weights)
        num_samples = nnorms
        proxies = self._normalizer.copy()
        proxies._save_name = save_name
        proxies.importance = jnp.exp(
            self._beta * imp_weights - logsumexp(self._beta * imp_weights)
        )

        visualize_mcmc_feature(
            chains=[proxies.weights],
            rates=[1.0],
            fig_dir=f"{self._save_dir}/mcmc",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples:04d}",
            log_scale=False,
            hist_weights=proxies.importance,
            **self._weight_params,
        )
        # Visualize distribution per feature pairs
        visualize_mcmc_pairs(
            chains=[proxies.weights],
            fig_dir=f"{self._save_dir}/mcmc/pairs",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples}_pairs",
            log_scale=False,
            **self._weight_params,
        )
        proxies.visualize(true_w=self.true_w, obs_w=None)
        proxies.save()

        return proxies

    def simulate_mcmc(self, tasks, save_name, tqdm_position=0, universal_model=None):
        """Sample 1 set of weights from b(design_w) given true_w and prior tasks.

        Example:
            >>> designer.simulate(task, "designer_task_1")

        Args:
            tasks: new environment task
                shape (n, task_dim)
            save_name (save_name): name for saving parameters and visualizations
            universal_model: trained network to predict expected feature counts

        """
        print(f"Sampling Designer (prior={len(self._prior_tasks)}): {save_name}")
        if self._use_true_w:
            particles = self.create_particles(
                [self.true_w] * self._sample_args["num_samples"],
                controller=self._one_controller,
                runner=self._one_runner,
                save_name=save_name,
            )
            return particles

        ## Sample based on prior_tasks + tasks
        assert self._truth is not None, "Need assumed designer truth."
        assert self._prior_tasks is not None, "Need >=0 prior tasks."

        ## Independent design mode
        print("tasks shape", tasks.shape)
        assert len(tasks.shape) == 2
        if self._design_mode == "independent":
            tasks = [tasks[-1]]
        if len(self._prior_tasks) > 0:
            tasks = onp.concatenate([self._prior_tasks, tasks])

        ## ==============================================================
        ## ======================= MCMC Sampling ========================
        init_args = copy.deepcopy(self._sample_init_args)
        samp_args = copy.deepcopy(self._sample_args)
        ntasks = len(tasks)
        decay = self._proposal_decay ** (ntasks - 1)
        init_args["proposal_var"] = init_args["proposal_var"] * decay
        num_chains = self._sample_args["num_chains"]
        num_keys = len(self._prior.feature_keys)
        init_params = DictList(
            dict(zip(self._prior.feature_keys, jnp.zeros((num_keys, 1))))
        )
        if num_chains > 1:
            #  shape nfeats * (nchains, 1)
            init_params = init_params.expand_dims(0).repeat(num_chains, axis=0)
        if self._normalized_key in init_params:
            del init_params[self._normalized_key]

        model = self._build_model(tasks, universal_model=universal_model)
        sampler = get_designer_sampler(self._sample_method, model, init_args, samp_args)

        self._rng_key, rng_sampler = random.split(self._rng_key, 2)
        sampler.run(
            rng_sampler,
            init_params=dict(init_params),
            extra_fields=["mean_accept_prob"],
            tqdm_position=tqdm_position,
        )

        ## ==============================================================
        ## ====================== Analyze Samples =======================
        sample_ws = sampler.get_samples(group_by_chain=True)
        #  shape nfeats * (nchains, nsample, 1) -> nfeats * (nchains, nsample)
        sample_ws = DictList(sample_ws).squeeze(axis=2)
        assert self._normalized_key not in sample_ws.keys()
        if self._normalized_key in sample_ws:
            sample_ws[self._normalized_key] = jnp.zeros(sample_ws.shape)
        sample_info = sampler.get_extra_fields(group_by_chain=True)
        sample_rates = sample_info["mean_accept_prob"][:, -1]
        num_samples = sample_ws.shape[1]

        visualize_mcmc_feature(
            chains=sample_ws,
            rates=sample_rates,
            fig_dir=f"{self._save_dir}/mcmc",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples:04d}",
            **self._weight_params,
        )
        # Visualize distribution per feature pairs
        visualize_mcmc_pairs(
            chains=sample_ws,
            fig_dir=f"{self._save_dir}/mcmc/pairs",
            title=f"seed_{self._rng_name}_{save_name}_samples_{num_samples}_pairs",
            **self._weight_params,
        )
        particles = self.create_particles(
            sample_ws[0].exp(),  # first chain
            controller=self._one_controller,
            runner=self._one_runner,
            save_name=save_name,
        )
        particles.visualize(true_w=self.true_w, obs_w=None)
        particles.save()

        # if self._select_mode == "map":
        #     return particles.map_estimate(1)
        # elif self._select_mode == "mean":
        #     return particles.subsample(1)
        # else:
        #     raise NotImplementedError
        return particles

    def _build_model(self, tasks, universal_model=None):
        """Build Designer PGM model."""

        ## Fill in unused feature keys
        assert self._likelihood is not None
        feats_keys = self._env.features_keys

        nchain = self._sample_args["num_chains"]
        nfeats = len(feats_keys)
        nnorms = len(self._normalizer.weights)
        ntasks = len(tasks)
        prior_keys = (
            self._prior_keys
            if self._prior_keys is not None
            else list(self._truth.weights.keys())
        )
        self.update_prior(prior_keys)
        #  shape (nfeats, 1)
        true_ws = self._truth.weights.prepare(feats_keys).numpy_array()
        assert true_ws.shape == (nfeats, 1)
        ## Operation over tasks
        task_method = None
        if self._task_method == "sum":
            task_method = partial(jnp.sum, axis=0)
        elif self._task_method == "mean":
            task_method = partial(jnp.mean, axis=0)
        else:
            raise NotImplementedError

        if universal_model is None:
            ## ============= Pre-empt optimiations =============
            #  shape (nfeats, ntasks, 1)
            true_fsum = self._truth.get_features_sum(tasks).prepare(feats_keys)
            true_fsum = true_fsum.numpy_array()

            def _model():
                #  shape nfeats * (1,)
                new_ws = self._prior(1)
                log_prior = jnp.log(new_ws.numpy_array())
                infeasible = jnp.logical_or(
                    jnp.any(log_prior < -1 * self._max_val),
                    jnp.any(log_prior > self._max_val),
                )

                new_ws = new_ws.prepare(feats_keys).normalize_across_keys()
                sample_ws = new_ws.numpy_array()
                ## ======= Not jit-able optimization: requires scipy/jax optimizer ======
                sample_ps = self.create_particles(
                    new_ws,
                    controller=self._sample_controller,
                    runner=self._sample_runner,
                    jax=True,
                )
                sample_ps.compute_tasks(tasks, jax=True)
                #  shape (nfeats, ntasks, 1)
                sample_feats_sum = sample_ps.get_features_sum(tasks).prepare(feats_keys)
                sample_feats_sum = sample_feats_sum.numpy_array()
                assert sample_feats_sum.shape == (nfeats, ntasks, 1)
                #  shape (nbatch=1,)
                true_offset = jnp.zeros(1)
                #  shape (ntasks, nbatch=1,)
                log_probs = self._likelihood(
                    true_ws, true_offset, sample_ws, sample_feats_sum, tasks, self.beta
                )
                #  shape (nbatch=1,)
                log_prob = jnp.where(infeasible, -1e10, task_method(log_probs))
                numpyro.factor("designer_log_prob", log_prob)

            return _model
        else:
            import torch

            def _model():
                #  shape nfeats * (1,)
                new_ws = self._prior(1)
                sample_ws = new_ws.prepare(feats_keys).onp_array()
                #  shape (1, prior nfeats)
                sample_input = (
                    torch.from_numpy(new_ws.onp_array()).float().permute(1, 0)
                )
                # TODO: currently only solves for one task
                #  shape (1, nfeats)
                sample_feats_sum = universal_model(sample_input)
                sample_feats_sum = jnp.array(
                    sample_feats_sum.detach().numpy()
                ).swapaxes(1, 0)[:, None]
                assert sample_feats_sum.shape == (nfeats, ntasks, 1)
                #  shape (nbatch=1,)
                true_offset = jnp.zeros(1)
                #  shape (ntasks, nbatch=1,)
                log_probs = self._likelihood(
                    true_ws, true_offset, sample_ws, sample_feats_sum, tasks, self.beta
                )
                #  shape (nbatch=1,)
                log_prob = task_method(log_probs)
                numpyro.factor("designer_log_prob", log_prob)

            return _model

    def _build_likelihood(self):
        """Build likelihood kernel."""

        ## Operation over tasks
        task_method = None
        if self._task_method == "sum":
            task_method = partial(jnp.sum, axis=0)
        elif self._task_method == "mean":
            task_method = partial(jnp.mean, axis=0)
        else:
            raise NotImplementedError

        @jax.jit
        def _likelihood(true_ws, true_offset, sample_ws, sample_feats_sum, tasks, beta):
            """
            Forward likelihood function (Unnormalized). Used in Designer sampling.
            Computes something proportional to p(design_w | true_w).

            Args:
                true_ws (ndarray): designer's true w in mind
                    shape: (nfeats, nbatch)
                true_offset (ndarray): sample offset
                    shape: (nbatch,)
                tasks (ndarray): prior tasks
                    shape: (ntasks, nbatch, task_dim)
                sample_ws (ndarray): current sampled w, batched
                    shape: (nfeats, nbatch)
                sample_feats_sum (ndarray): sample features
                    shape: (nfeats, ntasks, nbatch)

            Note:
                * nbatch dimension has two uses
                  (1) nbatch=1 in designer.sample
                  (1) nbatch=nobs, ntasks=1 in ird.sample
                * To stabilize MH sampling
                  (1) sum across tasks
                  (2) sum across features

            Return:
                log_probs (ndarray): (ntasks, nbatch, )

            """
            nfeats = sample_ws.shape[0]
            nbatch = sample_ws.shape[1]
            ntasks = len(tasks)
            assert true_ws.shape == (nfeats, nbatch)
            assert true_offset.shape == (nbatch,)
            assert sample_ws.shape == (nfeats, nbatch)
            assert sample_feats_sum.shape == (nfeats, ntasks, nbatch)

            #  shape (nfeats, 1, nbatch)
            true_ws = jnp.expand_dims(true_ws, axis=1)

            ## ===============================================
            ## ======= Computing Numerator: sample_xxx =======
            #  shape (nfeats, ntasks, nbatch)
            sample_costs = true_ws * sample_feats_sum
            #  shape (nfeats, ntasks, nbatch) -> (ntasks, nbatch, ), sum across features
            sample_costs = sample_costs.sum(axis=0) + true_offset
            sample_rews = -beta * sample_costs
            # sample_costs = sample_costs.mean(axis=0)
            assert sample_rews.shape == (ntasks, nbatch)
            #  shape (ntasks, nbatch,)
            log_probs = sample_rews
            return log_probs

        def _normalized_likelihood(
            true_ws,
            true_offset,
            true_feats_sum,
            sample_ws,
            normal_feats_sum,
            tasks,
            beta,
        ):
            """
            Main likelihood function (Normalized). Used in Designer Inversion (IRD Kernel).
            Approximates p(design_w | true_w).

            Args:
                true_offset (ndarray): true ws offset
                    shape: (nbatch,)
                normal_feats_sum (ndarray): normalizer features
                    shape: (nfeats, ntasks, nbatch, nnorms)
                true_feats_sum (ndarray): true ws features
                    shape: (nfeats, ntasks, nbatch)

            Note:
                * In IRD kernel, `obs_feats_sum` is used as proxy for `true_feats_sum`
                * For performance, nbatch & nnorms can be huge in practice
                  nbatch * nnorms ~ 2000 * 200 in IRD kernel
                * nbatch dimension has two uses
                  (1) nbatch=1 in designer.sample
                  (1) nbatch=nobs, ntasks=1 in ird.sample
                * To stabilize MH sampling
                  (1) sum across tasks
                  (2) sum across features

            Return:
                log_probs (ndarray): (nbatch, )

            """
            nfeats = sample_ws.shape[0]
            nbatch = sample_ws.shape[1]
            ntasks = len(tasks)
            nnorms = normal_feats_sum.shape[3]
            assert true_ws.shape == (nfeats, nbatch)  # same, repeated `nbatch` times
            assert sample_ws.shape == (nfeats, nbatch)
            assert true_feats_sum.shape == (nfeats, ntasks, nbatch)
            assert normal_feats_sum.shape == (nfeats, ntasks, nbatch, nnorms)

            #  shape (ntasks, nbatch)
            sample_rews = _likelihood(
                true_ws, true_offset, sample_ws, true_feats_sum, tasks, beta
            )

            ## =================================================
            ## ======= Computing Denominator: normal_xxx =======
            #  shape (nfeats, 1, nbatch, 1)
            normal_truth = jnp.expand_dims(jnp.expand_dims(true_ws, axis=1), 3)
            #  shape (nfeats, ntasks, nbatch, nnorms + 1)
            normal_feats_sum = jnp.concatenate(
                [normal_feats_sum, jnp.expand_dims(true_feats_sum, axis=3)], axis=3
            )
            #  shape (nfeats, ntasks, nbatch, nnorms + 1)
            normal_costs = normal_truth * normal_feats_sum
            #  shape (ntasks, nbatch, nnorms + 1)
            normal_costs = normal_costs.sum(axis=0) + true_offset[:, None]
            normal_rews = -beta * normal_costs
            # normal_rews = -beta * normal_costs.mean(axis=0)

            ## Normalize by tasks
            # normal_rews -= jnp.expand_dims(sample_rews, axis=2)
            # sample_rews = jnp.zeros_like(sample_rews)

            #  shape (nbatch, nnorms + 1)
            normal_rews = task_method(normal_rews)
            #  shape (nbatch,)
            sample_rews = task_method(sample_rews)
            #  shape (nbatch,)
            normal_rews = logsumexp(normal_rews, axis=1) - jnp.log(nnorms + 1)
            assert normal_rews.shape == (nbatch,)

            ## =================================================
            ## ============== Computing Probs ==================
            #  shape (nbatch,)
            log_probs = sample_rews - normal_rews
            return log_probs

        return _likelihood, _normalized_likelihood

    def create_particles(
        self, weights, controller=None, runner=None, save_name="", jax=False
    ):
        weights = DictList(weights, jax=jax)
        if controller is None:
            controller = self._controller
        if runner is None:
            runner = self._runner
        self._rng_key, rng_particles = random.split(self._rng_key, 2)
        return Particles(
            rng_name=self._rng_name,
            rng_key=rng_particles,
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


class DesignerInteractive(Designer):
    """Interactive Designer used in Jupyter Notebook interactive mode.

    Args:
        normalized_key (str): used to normalize user-input, e.g. keep "dist_cars" weight 1.0.

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller_fn,
        name="Default",
        save_dir="",
        normalized_key=None,
    ):
        super().__init__(
            rng_key=rng_key,
            env_fn=env_fn,
            controller_fn=controller_fn,
            beta=None,
            truth=None,
            prior_fn=None,
            sample_method=None,
            save_dir=save_dir,
            exp_name="interactive",
        )
        self._name = name
        self._save_dir = f"{self._save_dir}/{self._exp_name}/{save_name}"
        os.makedirs(self._save_dir)
        self._user_inputs = []
        self._normalized_key = normalized_key

    def sample(self, task, verbose=True, itr=0):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def simulate(self, task, save_name):
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
            f"{self._save_dir}/key_{self._rng_name}_user_trial_0_task_{str(task)}.png"
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
                acs = self._controller(init_state, user_in_w)
                num_weights = len(self._user_inputs)
                video_path = f"{self._save_dir}/key_{self._rng_name}_user_trial_{num_weights}_task_{str(task)}.mp4"
                print("Received Weights")
                pp.pprint(user_in_w)
                self._runner.nb_show_mp4(init_state, acs, path=video_path, clear=False)
            except Exception as e:
                print(e)
                print("Invalid input.")
                continue
        user_w = DictList([self._user_inputs[-1]])
        self._rng_key, rng_particles = random.split(self._rng_key, 2)
        return Particles(
            rng_name=self._rng_name,
            rng_key=rng_particles,
            env_fn=self._env_fn,
            controller=self._one_controller,
            runner=self._one_runner,
            weights=user_w,
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

    def _build_model(self, beta):
        """Dummy model."""
        pass
