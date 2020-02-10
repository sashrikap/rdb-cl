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
from rdb.infer.utils import *
from rdb.infer.pgm import PGM
from numpyro.handlers import seed
from tqdm.auto import tqdm, trange
from rdb.infer.dictlist import DictList
from rdb.infer.particles import Particles


class Designer(PGM):
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
        prior,
        proposal,
        normalized_key,
        num_normalizers,
        sample_method="mh",
        sampler_args={},
        save_root="",
        exp_name="",
        use_true_w=False,
        weight_params={},
        num_prior_tasks=0,
    ):
        self._rng_key = rng_key
        self._env_fn = env_fn
        self._env = env_fn()
        self._build_controllers(controller_fn)
        self._true_w = true_w
        if self._true_w is not None:
            self._truth = self.create_particles(
                weights=[true_w],
                controller=self._one_controller,
                runner=self._one_runner,
                save_name="designer_truth",
            )
        else:
            self._truth = None
        self._use_true_w = use_true_w
        self._normalized_key = normalized_key
        self._num_normalizers = num_normalizers
        self._normalizer = None
        # Sampling Prior and kernel
        self._prior = prior
        self._kernel = self._build_kernel(beta)
        # Cache
        super().__init__(
            rng_key, self._kernel, prior, proposal, sample_method, sampler_args
        )
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        self._weight_params = weight_params
        # Designer Prior
        self._random_choice = None
        self._prior_tasks = []
        self._num_prior_tasks = num_prior_tasks

    def _build_controllers(self, controller_fn):
        self._controller, self._runner = controller_fn(self._env, "Designer Core")
        self._norm_controller, self._norm_runner = controller_fn(
            self._env, "Designer Norm"
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

            ## Preemp computation
            self._normalizer.compute_tasks(tasks, vectorize=False)
            ## Sample
            sample_ws, info = self._sampler.sample(
                obs=self._truth.weights[0], tasks=tasks, init_state=None, name=save_name
            )
            ## Visualize multiple MCMC chains to check convergence.
            visualize_chains(
                chains=info["all_chains"],
                rates=info["rates"],
                fig_dir=f"{self._save_dir}/mcmc",
                title=save_name,
                **self._weight_params,
            )
            sample_ws = DictList(sample_ws)
            samples = self.create_particles(
                sample_ws,
                controller=self._one_controller,
                runner=self._one_runner,
                save_name=save_name,
            )
            samples.visualize(true_w=self.true_w, obs_w=None)
            return samples.subsample(1)

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
            weights=[w],
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

    def update_key(self, rng_key):
        super().update_key(rng_key)
        if self._truth is not None:
            self._truth.update_key(rng_key)
        self._sampler.update_key(rng_key)
        self._prior.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        self.reset_prior_tasks()
        norm_ws = self._prior.sample(self._num_normalizers)
        self._normalizer = self.create_particles(
            norm_ws,
            save_name="designer_normalizer",
            runner=self._norm_runner,
            controller=self._norm_controller,
        )

    def _build_kernel(self, beta):
        @partial(jax.jit, static_argnums=(2,))
        def _batch_fn(batch_arr, normal_arr, out_shape):
            """Multiply, average and logsumexp two very costly array in likelihood_fn.

            Args:
                batch_arr (ndarray): (nfeats, 1, nbatch, 1)
                normal_arr (ndarray): (nfeats, ntasks, 1, nnorms)
                out_shape (tuple): (ntasks, nbatch, nnorms)

            """

            def _mult_add(i, sum_):
                return sum_ + np.multiply(batch_arr[i], normal_arr[i])

            assert len(batch_arr.shape) == len(normal_arr.shape) == 4
            # return batch_arr * normal_arr
            nfeats = len(batch_arr)
            sum_ = np.zeros(out_shape)
            sum_ = jax.lax.fori_loop(0, nfeats, _mult_add, sum_)
            mean_arr = sum_ / nfeats
            return np_logsumexp(-beta * mean_arr, axis=2)

        def likelihood_fn(true_ws, sample_ws, tasks, sample=None):
            """Designer forward likelihood p(design_w | true_w).

            Args:
                true_ws (DictList): designer's true w in mind
                    shape: nfeats * (nbatch, )
                sample_ws (DictList(nbatch)): current sampled w, batched
                    shape: nfeats * (nbatch, )
                tasks (ndarray): prior tasks
                    shape: (ntasks, task_dim)
                sample (Particles): if provided, Particles(sample_ws) with injected experience (speed-up)

            Note:
                * For performance, nbatch & nnorms can be huge in practice
                * nbatch dimension has two uses
                  (1) nbatch=nchain in designer.sample
                  (1) nbatch=nnorms in ird.sample
                * To stabilize MH sampling
                  (1) average across tasks (instead of sum)
                  (2) average across features costs (instead of sum)

            Return:
                log_probs (ndarray): (nbatch, )

            """

            normal = self._normalizer
            nbatch = len(sample_ws)
            ntasks = len(tasks)
            nnorms = len(normal.weights)
            # nfeats = sample_ws.num_keys
            nfeats = len(self._env.features_keys)
            assert true_ws.shape == (nbatch,)
            if sample is None:
                sample = self.create_particles(
                    weights=sample_ws,
                    controller=self._sample_controller,
                    runner=self._sample_runner,
                    save_name="designer_sample",
                )
            else:
                assert sample.weights.shape == sample_ws.shape
            #  shape nfeats * (1, nbatch)
            truth = true_ws.expand_dims(axis=0).prepare(self._env.features_keys)

            ## Computing Numerator: sample_xxx
            #  shape nfeats * (ntasks, nbatch)
            sample_feats_sum = sample.get_features_sum(tasks)
            #  shape (nfeats, ntasks, nbatch)
            sample_costs = (truth * sample_feats_sum).onp_array()
            assert sample_costs.shape == (nfeats, ntasks, nbatch)
            #  shape (nfeats, ntasks, nbatch) -> (nbatch, ), average across features and tasks
            sample_rews = (-beta * sample_costs).mean(axis=(0, 1))
            assert sample_rews.shape == (nbatch,)

            ## Computing Denominator: normal_xxx
            #  shape nfeats * (ntasks, nnorms)
            normal_feats_sum = normal.get_features_sum(tasks)
            #  shape (nfeats, ntasks, 1, nnorms)
            normal_feats_sum = normal_feats_sum.expand_dims(1).numpy_array()
            #  shape (nfeats, 1, nbatch, 1)
            normal_truth = truth.expand_dims(2).numpy_array()
            #  shape (ntasks, nbatch)
            normal_rews = _batch_fn(
                normal_truth, normal_feats_sum, (ntasks, nbatch, nnorms)
            )
            assert normal_rews.shape == (ntasks, nbatch)
            #  shape (ntasks, nbatch, nnorms) -> (ntasks, nbatch) across features & norms
            # normal_rews = logsumexp(-beta * normal_costs, axis=2)
            normal_rews -= onp.log(nnorms)
            assert normal_rews.shape == (ntasks, nbatch)

            ## Computing Probs
            #  shape (ntasks, nbatch)
            log_probs = sample_rews[None, :] - normal_rews
            #  shape (ntasks, nbatch) -> (nbatch,)
            log_probs = log_probs.mean(axis=0)
            # if True:
            if False:
                print(f"Designer prob {log_probs:.3f}")
            return log_probs

        return likelihood_fn


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

    def _build_sampler(self, kernel, proposal, sample_method, sample_args):
        """Dummy sampler."""
        pass

    def _build_kernel(self, beta):
        """Dummy kernel."""
        pass
