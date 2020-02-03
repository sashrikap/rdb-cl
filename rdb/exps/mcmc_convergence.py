"""Experiment to check the convergence of MCMC algorithms.

Observation:
    * MCMC process may vary under different conditions e.g.
number of observations in IRD and number of prior tasks in designer model

Goal:
    * Find # samples it needs to converge.

Includes:
    * testing ird convergence `exp.run_ird()`
    * testing designer convergence `exp.run_designer()`

"""

from rdb.infer.utils import random_choice
from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from tqdm.auto import tqdm
from time import time
import jax.numpy as np
import numpy as onp
import copy
import os


class ExperimentMCMC(object):
    """MCMC Convergence experiment.

    """

    def __init__(
        self,
        model,
        eval_server,
        num_eval_tasks=4,
        num_eval_map=-1,
        fixed_task_seed=None,
        normalized_key=None,
        design_data={},
        num_load_design=-1,
        save_root="data/test",
        exp_name="mcmc_convergence",
        exp_params={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        self._random_choice = None
        self._random_task_choice = None
        self._normalized_key = normalized_key
        # Evaluation
        self._num_eval_map = num_eval_map
        self._num_eval_tasks = num_eval_tasks
        # Active Task proposal
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._last_time = time()
        # Load design and cache
        self._num_load_design = num_load_design
        self._design_data = design_data

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._designer.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        if self._fixed_task_seed is not None:
            self._random_task_choice = seed(random_choice, self._fixed_task_seed)
        else:
            self._random_task_choice = self._random_choice

    def run_designer(self, num_prior_tasks, prior_mode, num_design_tasks):
        """Simulate designer on new_task. Varying the number of latent
        tasks as prior

        Args:
            num_prior_tasks (int): how many prior tasks to load
            prior_mode (str): "design" or "random"
            num_design_tasks (int): if > 1, then design a few tasks simultaneously to speed up

        """
        print(f"Prior task number: {num_prior_tasks}")
        self._designer.prior_tasks = self._load_tasks(num_prior_tasks, mode)

        ## Find evaluation tasks
        assert self._random_task_choice is not None
        all_tasks = self._designer.env.all_tasks
        ## Simulate
        design_tasks = self._random_task_choice(all_tasks, num_design_tasks)
        design_task_names = [f"designer_{task}" for task in design_tasks]
        obs = self._designer.simulate(
            task,
            task_name,
            save_name=f"designer_seed_{str(self._rng_key)}_prior_{num_prior_tasks}",
        )

        # ## Evaluate
        # num_eval = min(self._num_eval_tasks, len(self._designer.env.all_tasks))
        # all_tasks = self._random_task_choice(
        #     self._designer.env.all_tasks, num_eval, replacement=False
        # )
        # all_names = [f"eval_{task}" for task in all_tasks]
        # self._eval_server.compute_tasks(obs, all_tasks, all_names, verbose=True)
        # truth = self._designer.truth
        # self._eval_server.compute_tasks(truth, all_tasks, all_names, verbose=True)

        # ## Visualize performance
        # all_diff_rews = []
        # for task_, name_ in zip(all_tasks, all_names):
        #     diff_rews, _ = obs.compare_with(task_, name_, truth)
        #     all_diff_rews.append(diff_rews[0])
        # plot_dir = f"{self._save_dir}/plots"
        # os.makedirs(plot_dir, exist_ok=True)
        # fig_path = f"{plot_dir}/designer_seed_{str(self._rng_key)}"
        # fig_suffix = f"_prior_{num_prior_tasks:02d}"
        # obs.visualize_tasks(fig_path, fig_suffix, all_names, all_diff_rews)

        ## Reset designer prior tasks
        self._designer.prior_tasks = all_tasks

    def run_ird(self, num_obs, mode):
        """Run IRD on task, varying the number of past observations."""
        print(f"Observation number: {num_obs}")
        observations = self._load_observations(num_obs, mode)

        ## Find evaluation tasks
        assert self._random_task_choice is not None
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(self._designer.env.all_tasks):
            num_eval = -1
        all_tasks = self._random_task_choice(
            self._designer.env.all_tasks, num_eval, replacement=False
        )
        all_names = [f"eval_{task}" for task in all_tasks]

        ## Simulate
        task_name = f"designer_{task}"
        obs = self._designer.simulate(
            task,
            task_name,
            save_name=f"designer_seed_{str(self._rng_key)}_prior_{num_prior_tasks}",
        )

        ## Evaluate
        self._eval_server.compute_tasks(obs, all_tasks, all_names, verbose=True)
        truth = self._designer.truth
        self._eval_server.compute_tasks(truth, all_tasks, all_names, verbose=True)

        # ## Visualize performance
        # all_diff_rews = []
        # for task_, name_ in zip(all_tasks, all_names):
        #     diff_rews, _ = obs.compare_with(task_, name_, truth)
        #     all_diff_rews.append(diff_rews[0])
        # plot_dir = f"{self._save_dir}/plots"
        # os.makedirs(plot_dir, exist_ok=True)
        # fig_path = f"{plot_dir}/designer_seed_{str(self._rng_key)}"
        # fig_suffix = f"_prior_{num_prior_tasks:02d}"
        # obs.visualize_tasks(fig_path, fig_suffix, all_names, all_diff_rews)

        ## Reset designer prior tasks
        self._designer.prior_tasks = all_tasks

    def _load_tasks(self, mode):
        if mode == "random":
            pass
        elif mdoe == "design":
            pass
        else:
            raise NotImplementedError

    def _load_observations(self, mode):
        if mode == "random":
            pass
        elif mdoe == "design":
            pass
        else:
            raise NotImplementedError
