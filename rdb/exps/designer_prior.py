"""Informed Designer Experiment.

See how we can construct the best prior knowledge for reward designer.

Full Informed Designer Experiment:
    1)

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


class ExperimentDesignerPrior(object):
    """Informed Designer Experiment.

    Args:
        designer (object): PGM-based reward designer
        eval_server (object)
        num_prior_tasks (int): number of latent tasks

    """

    def __init__(
        self, rng_key, designer, eval_server, num_eval_tasks, save_dir, exp_name
    ):
        self._rng_key = rng_key
        self._designer = designer
        self._eval_server = eval_server
        self._num_eval_tasks = num_eval_tasks
        # For designer
        self._max_prior_tasks = 8
        self._random_choice = None
        # For checkpointing
        self._save_dir = save_dir
        self._exp_name = exp_name

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._designer.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)

    def run(self, task, num_prior_tasks):
        """Simulate designer on `task`. Varying the number of latent
        tasks as prior
        """
        assert num_prior_tasks < self._max_prior_tasks
        print(f"Prior task number: {num_prior_tasks}")

        ## Change prior tasks
        all_tasks = copy.deepcopy(self._designer.prior_tasks)
        self._designer.prior_tasks = all_tasks[:num_prior_tasks]

        ## Find evaluation tasks
        assert self._random_choice is not None
        all_tasks = self._random_choice(
            self._designer.env.all_tasks, self._num_eval_tasks, replacement=False
        )
        all_names = [f"eval_{task}" for task in all_tasks]

        ## Simulate & Evaluate
        task_name = f"designer_{task}"
        obs = self._designer.simulate(task, task_name)
        self._eval_server.compute_tasks(obs, all_tasks, all_names, verbose=True)
        truth = self._designer.truth
        self._eval_server.compute_tasks(truth, all_tasks, all_names, verbose=True)

        ## Visualize performance
        all_diff_rews = []
        for task_, name_ in zip(all_tasks, all_names):
            diff_rews, _ = obs.compare_with(task_, name_, truth)
            all_diff_rews.append(diff_rews[0])
        plot_dir = f"{self._save_dir}/{self._exp_name}/plots"
        os.makedirs(plot_dir, exist_ok=True)
        fig_path = f"{plot_dir}/designer_seed_{str(self._rng_key)}"
        fig_suffix = f"_prior_{num_prior_tasks:02d}"
        obs.visualize_tasks(fig_path, fig_suffix, all_names, all_diff_rews)

        ## Reset designer prior tasks
        self._designer.prior_tasks = all_tasks
