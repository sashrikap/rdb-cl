"""Informed Designer Experiment.

See how we can construct the best prior knowledge for reward designer.

Full Informed Designer Experiment:
    1)

"""

from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from tqdm.auto import tqdm
import jax.numpy as np
import numpy as onp
import time
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
        self,
        rng_key,
        designer,
        eval_server,
        num_eval_tasks,
        save_root,
        exp_name,
        fixed_task_seed=None,
    ):
        self._rng_key = rng_key
        self._designer = designer
        self._eval_server = eval_server
        self._num_eval_tasks = num_eval_tasks
        # For designer
        self._max_prior_tasks = 8
        self._random_task_choice = None
        self._fixed_task_seed = fixed_task_seed
        # For checkpointing
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._designer.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        if self._fixed_task_seed is not None:
            self._random_task_choice = seed(random_choice, self._fixed_task_seed)
        else:
            self._random_task_choice = self._random_choice

    def run(self, task, num_prior_tasks, evaluate=True):
        """Simulate designer on `task`. Varying the number of latent
        tasks as prior
        """
        assert num_prior_tasks < self._max_prior_tasks
        print(f"Prior task number: {num_prior_tasks}")

        ## Change prior tasks
        all_tasks = copy.deepcopy(self._designer.prior_tasks)
        self._designer.prior_tasks = all_tasks[:num_prior_tasks]

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

        if evaluate:
            ## Evaluate
            self._eval_server.compute_tasks(obs, all_tasks, all_names, verbose=True)
            truth = self._designer.truth
            self._eval_server.compute_tasks(truth, all_tasks, all_names, verbose=True)

            ## Visualize performance
            all_diff_rews = []
            for task_, name_ in zip():
                diff_rews, _ = obs.compare_with(task_, name_, truth)
                all_diff_rews.append(diff_rews[0])
            plot_dir = f"{self._save_dir}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            fig_name = f"prior_{num_prior_tasks:02d}"
            obs.visualize_comparisons(
                all_tasks, all_names, truth, fig_name, all_diff_rews
            )

        ## Reset designer prior tasks
        self._designer.prior_tasks = all_tasks
