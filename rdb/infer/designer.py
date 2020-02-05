import pprint
import copy
import json
import os
import jax
import jax.numpy as np
import numpyro, itertools
import numpy as onp
from rdb.optim.utils import multiply_dict_by_keys, append_dict_by_keys
from numpyro.handlers import scale, condition, seed
from rdb.infer.utils import random_choice, logsumexp
from rdb.infer.particles import Particles
from rdb.infer.pgm import PGM
from tqdm.auto import tqdm, trange
from rdb.exps.utils import *
from os.path import join
from rdb.infer.algos import *
from rdb.infer.utils import *
from time import time


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
        super().__init__(rng_key, self._kernel, proposal, sample_method, sampler_args)
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        self._weight_params = weight_params
        # Designer Prior
        self._random_choice = None
        self._prior_tasks = []
        self._num_prior_tasks = num_prior_tasks

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

    def simulate(self, new_task, task_name, save_name):
        """Sample 1 set of weights from b(design_w) given true_w.

        Args:
            new_task: new environment task
            task_name (str):
            save_name (save_name): name for saving parameters and visualizations

        """
        print(f"Sampling Designer (prior={len(self._prior_tasks)}): {save_name}")
        if self._use_true_w:
            sample_ws = onp.array([self._true_w])
        else:
            # Sample based on prior tasks + new_task
            sample_tasks = onp.concatenate([self._prior_tasks, new_task])
            sample_ws, info = self._sampler.sample(
                obs=self._true_w, tasks=sample_tasks, name=save_name
            )
        import pdb

        pdb.set_trace()
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
        samples.visualize(true_w=self.true_w, obs_w=None)
        # Visualize multiple MCMC chains to check convergence.
        visualize_chains(
            info["all_chains"],
            fig_dir=f"{self._save_dir}/mcmc",
            title=save_name,
            **self._weight_params,
        )

        self._visualize_chains(save_name=save_name),
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
        self._true_w = w

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
        def likelihood_fn(true_w, sample_w, tasks):
            """Designer forward likelihood p(design_w | true_w).

            Args:
                true_w (DictList): designer's true w in mind
                    shape: (nweight, 1)
                sample_w (DictList): current sampled w
                    shape: (nweight, nbatch, 1)
                tasks (ndarray): prior tasks
                    shape: (ntasks, task_dim)

            """
            assert self._prior_tasks is not None, "Need >=0 prior tasks."
            assert len(tasks) == self._num_prior_tasks + 1
            nbatch = len(sample_w)
            ntasks = len(tasks)
            ## Do something on each sample each task
            ## Cross product:
            ##  (nbatch,) x (ntasks,) -> (nbatch * ntasks,)
            pairs = list(itertools.product(sample_w, tasks))
            batch_ws, batch_tasks = zip(*pairs)
            batch_tasks = np.array(batch_tasks)
            batch_ws = DictList(batch_ws)
            true_ws = DictList(true_w, expand_dims=True).repeat(nbatch * ntasks, axis=0)

            ## Calculate probability
            log_probs = onp.zeros(nbatch * ntasks)
            states = self.env.get_init_states(batch_tasks)
            actions = self._controller(states, batch_ws)
            _, costs, info = self._runner(states, actions, weights=true_ws)
            rews = -1 * costs
            log_probs += beta * rews
            log_probs += self._prior.log_prob(batch_ws)
            # if True:
            if False:
                print(f"Designer prob {log_prob:.3f}", actions.mean())
            return log_probs

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
