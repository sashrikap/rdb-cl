import os
import jax
import copy
import json
import pprint
import numpy as onp
import jax.numpy as np
import numpyro, itertools
from time import time
from os.path import join
from rdb.exps.utils import *
from functools import partial
from rdb.infer.utils import *
from numpyro.handlers import seed
from tqdm.auto import tqdm, trange
from rdb.infer.dictlist import DictList
from rdb.infer.particles import Particles
from jax.scipy.special import logsumexp as jax_logsumexp


class Designer(object):
    """Simulated designer module.

    Given true weights, sample MAP design weights.

    Note:
        * Currently assumes same beta, proposal as IRD module, although
          doesn't have to be.

    Args:
        true_w (dict): true weight

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller_fn,
        beta,
        true_w,
        prior_fn,
        ## Weight parameters
        normalized_key,
        num_normalizers,
        ## Sampling
        sample_method="MH",
        sample_init_args={},
        sample_args={},
        ## Parameter for histogram
        weight_params={},
        ## Saving options
        save_root="",
        exp_name="",
        use_true_w=False,
        num_prior_tasks=0,
    ):
        self._rng_key = rng_key
        self._env_fn = env_fn
        self._env = env_fn()
        self._build_controllers(controller_fn)
        self._true_w = true_w
        if self._true_w is not None:
            self._truth = self.create_particles(
                weights=DictList([true_w], jax=False),
                controller=self._one_controller,
                runner=self._one_runner,
                save_name="designer_truth",
            )
        else:
            self._truth = None
        self._beta = beta
        self._use_true_w = use_true_w
        self._normalized_key = normalized_key
        self._weight_params = weight_params

        ## Normalizer
        self._num_normalizers = num_normalizers
        self._normalizer = None
        self._norm_prior = None

        # Sampling Prior and kernel
        self._prior_fn = prior_fn
        self._prior = None
        self._model = None
        self._sampler = None
        self._likelihood_ird = None
        self._likelihood_designer = None
        self._sample_args = sample_args
        self._sample_method = sample_method
        self._sample_init_args = sample_init_args

        ## Saving
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"

        ## Designer Prior tasks
        self._random_choice = None
        self._prior_tasks = []
        self._num_prior_tasks = num_prior_tasks

    @property
    def likelihood(self):
        return self._likelihood_designer

    @property
    def likelihood_ird(self):
        return self._likelihood_ird

    @property
    def normalizer(self):
        return self._normalizer

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

    def create_particles(self, weights, controller=None, runner=None, save_name=""):
        weights = DictList(weights)
        if controller is None:
            controller = self._controller
        if runner is None:
            runner = self._runner
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

    def simulate(self, new_tasks, save_name):
        """Sample 1 set of weights from b(design_w) given true_w and prior tasks.

        Example:
            >>> designer.simulate(task, "designer_task_1")

        Args:
            new_tasks: new environment task
                shape (1, task_dim)
            save_name (save_name): name for saving parameters and visualizations

        """
        print(f"Sampling Designer (prior={len(self._prior_tasks)}): {save_name}")
        if self._use_true_w:
            return self._truth
        else:
            assert self._prior_tasks is not None, "Need >=0 prior tasks."
            ## Sample based on prior tasks + new_tasks
            if len(self._prior_tasks) == 0:
                assert len(new_tasks.shape) == 2 and len(new_tasks) == 1
                tasks = new_tasks
            else:
                tasks = onp.concatenate([self._prior_tasks, new_tasks])

            ## Pre-empt computation
            self._normalizer.compute_tasks(tasks, vectorize=False)
            ## Sample
            true_ws = self._truth.weights
            self._sampler.run(
                self._rng_key,
                nbatch=1,
                true_ws=true_ws,
                tasks=tasks,
                init_params=true_ws[0],
            )
            samples = self._sampler.get_samples()
            ## Visualize multiple MCMC chains to check convergence.
            visualize_chains(
                chains=samples,
                rates=info["rates"],
                fig_dir=f"{self._save_dir}/mcmc",
                title=save_name,
                **self._weight_params,
            )
            sample_ws = DictList(sample_ws)
            particles = self.create_particles(
                sample_ws,
                controller=self._one_controller,
                runner=self._one_runner,
                save_name=save_name,
            )
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
        assert self._num_prior_tasks == len(self._prior_tasks)
        return self._prior_tasks

    @prior_tasks.setter
    def prior_tasks(self, tasks):
        """ Modifies Underlying Designer Prior. Use very carefully."""
        self._prior_tasks = tasks
        self._num_prior_tasks = len(tasks)

    @property
    def true_w(self):
        return self._true_w

    @true_w.setter
    def true_w(self, w):
        print("Designer truth updated")
        self._true_w = w
        self._truth = self.create_particles(
            weights=DictList([w], jax=False),
            controller=self._one_controller,
            runner=self._one_runner,
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

    def update_key(self, rng_key):
        self._rng_key = rng_key
        if self._truth is not None:
            self._truth.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        self.reset_prior_tasks()
        ## Sample normaling factor
        self._norm_prior = seed(self._prior_fn("designer_norm"), rng_key)
        self._normalizer = self.create_particles(
            self._norm_prior(self._num_normalizers),
            save_name="designer_normalizer",
            runner=self._norm_runner,
            controller=self._norm_controller,
        )
        # Build likelihood and model
        self._build_sampler()

    def _build_sampler(self):
        self._prior = self._prior_fn("designer")
        self._model = self._build_model(self._likelihood_designer)
        self._likelihood_designer = self._build_likelihood(self._beta)
        self._likelihood_ird = self._build_likelihood(self._beta)
        self._sampler = get_numpyro_sampler(
            self._sample_method, self._model, self._sample_init_args, self._sample_args
        )

    def _build_model(self, likelihood_fn):
        """Build Designer PGM model."""

        def _model(true_ws, nbatch, tasks):
            sample_ws = self._prior(nbatch)
            log_prob = likelihood_fn(
                true_ws.numpy_array(), sample_ws.numpy_array(), tasks
            )
            numpyro.factor("designer_log_prob", log_prob)

        return _model

    def _build_likelihood(self, beta):
        """Build likelihood kernel."""

        @jax.jit
        def _likelihood(true_ws, sample_ws, tasks, sample_feats_sum, normal_feats_sum):
            """
            Main likelihood function. Used in Designer and Designer Inversion (IRD Kernel).
            Designer forward likelihood p(design_w | true_w).

            Args:
                true_ws (ndarray): designer's true w in mind
                    shape: (nfeats, nbatch, )
                tasks (ndarray): prior tasks
                    shape: (ntasks, nbatch, task_dim)
                sample_ws (ndarray): current sampled w, batched
                    shape: (nfeats, nbatch, )
                sample_feats_sum (ndarray): sample features
                    shape: (nfeats, ntasks, nbatch)
                normal_feats_sum (ndarray): normalizer features
                    shape: (nfeats, ntasks, nbatch, nnorms)

            Note:
                * For performance, nbatch & nnorms can be huge in practice
                  nbatch * nnorms ~ 2000 * 200 in IRD kernel
                * nbatch dimension has two uses
                  (1) nbatch=nchain in designer.sample
                  (1) nbatch=nnorms in ird.sample
                * To stabilize MH sampling
                  (1) average across tasks (instead of sum)
                  (2) average across features costs (instead of sum)

            Return:
                log_probs (ndarray): (nbatch, )

            """
            nfeats = sample_ws.shape[0]
            nbatch = sample_ws.shape[1]
            ntasks = len(tasks)
            nnorms = normal_feats_sum.shape[3]
            assert true_ws.shape == (nfeats, nbatch)
            assert sample_ws.shape == (nfeats, nbatch)
            assert sample_feats_sum.shape == (nfeats, ntasks, nbatch)
            assert normal_feats_sum.shape == (nfeats, ntasks, nbatch, nnorms)

            #  shape (nfeats, 1, nbatch)
            true_ws = np.expand_dims(true_ws, axis=1)

            ## ===============================================
            ## ======= Computing Numerator: sample_xxx =======
            #  shape (nfeats, ntasks, nbatch)
            sample_costs = true_ws * sample_feats_sum
            #  shape (nfeats, ntasks, nbatch) -> (nbatch, ), average across features and tasks
            sample_rews = (-beta * sample_costs).mean(axis=(0, 1))
            assert sample_rews.shape == (nbatch,)

            ## =================================================
            ## ======= Computing Denominator: normal_xxx =======
            #  shape (nfeats, ntasks, nbatch, 1)
            normal_truth = np.expand_dims(true_ws.repeat(ntasks, axis=1), axis=3)
            #  shape (nfeats, ntasks, nbatch, nnorms + 1)
            normal_feats_sum = np.concatenate([normal_feats_sum, normal_truth], axis=3)
            #  shape (nbatch, nnorms + 1)
            normal_rews = (normal_truth * normal_feats_sum).mean(axis=(0, 1))
            #  shape (nbatch,)
            normal_rews = jax_logsumexp(normal_rews, axis=1) - np.log(nnorms + 1)
            assert normal_rews.shape == (nbatch,)

            ## =================================================
            ## ============== Computing Probs ==================
            #  shape (nbatch,)
            log_probs = sample_rews - normal_rews
            return log_probs

        return _likelihood


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
                acs = self._controller(init_state, user_in_w)
                num_weights = len(self._user_inputs)
                video_path = f"{self._save_dir}/key_{self._rng_key}_user_trial_{num_weights}_task_{str(task)}.mp4"
                print("Received Weights")
                pp.pprint(user_in_w)
                self._runner.nb_show_mp4(init_state, acs, path=video_path, clear=False)
            except Exception as e:
                print(e)
                print("Invalid input.")
                continue
        user_w = DictList([self._user_inputs[-1]])
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=self._controller,
            runner=self._runner,
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
