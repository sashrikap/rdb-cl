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

from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from functools import partial
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
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
        num_visualize_tasks=32,
        fixed_task_seed=None,
        normalized_key=None,
        save_root="data/test",
        exp_name="mcmc_convergence",
        exp_params={},
        exp_mode="",
        design_data={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        self._random_weight = None
        self._random_choice = None
        self._random_task_choice = None
        self._normalized_key = normalized_key
        self._fixed_task_seed = fixed_task_seed
        # Evaluation & Visualization
        self._num_eval_map = num_eval_map
        self._num_eval_tasks = num_eval_tasks
        self._num_visualize_tasks = num_visualize_tasks
        # Active Task proposal
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{save_root}/{exp_name}"
        self._last_time = time()
        # Load design and cache
        self._design_data = design_data
        self._exp_mode = exp_mode
        # Designer relevant data
        self._all_designer_prior_tasks = []
        self._all_designer_prior_ws = []
        self._designer = self._model.designer

    def _load_design(self):
        """Different mode of experiments.
        """
        all_tasks = self._model.env.all_tasks
        num_designs = len(self._design_data["DESIGNS"])
        for design in self._design_data["DESIGNS"]:
            design["WEIGHTS"] = normalize_weights(
                design["WEIGHTS"], self._normalized_key
            )

        if self._exp_mode.startswith("ird"):
            ## Testing IRD convergence, no need of these designs
            pass

        elif self._exp_mode == "designer_true_w_prior_tasks":
            ## Load well-defined Jerry's prior designs
            ## Test # design tasks vs convergence
            for d_data in self._design_data["DESIGNS"]:
                design_task = d_data["TASK"]
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_designer_prior_tasks.append(design_task)
                self._all_designer_prior_ws.append(true_w)

        elif self._exp_mode == "designer_true_w_random_tasks":
            ## Use one of Jerry's prior designed w as true w
            ## Test # random tasks vs convergence
            rand_tasks = self._random_task_choice(all_tasks, num_designs)
            self._all_designer_prior_tasks = rand_tasks
            for d_data in self._design_data["DESIGNS"]:
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_designer_prior_ws.append(true_w)

        elif self._exp_mode == "designer_random_w_random_tasks":
            ## Use a random w as true w
            ## Test # random tasks vs convergence
            rand_tasks = onp.array(self._random_task_choice(all_tasks, num_designs))
            self._all_designer_prior_tasks = rand_tasks
            rand_w = self._make_random_weights(
                self._design_data["DESIGNS"][-1]["WEIGHTS"].keys()
            )
            for d_data in self._design_data["DESIGNS"]:
                # rand_w = self._make_random_weights(d_data["WEIGHTS"].keys())
                self._all_designer_prior_ws.append(rand_w)

        elif self._exp_mode == "designer_true_w_more_features":
            ## Use a true w on random tasks
            ## The true w has progressively more random features
            ## Test # features (DOF) vs convergence
            all_keys = self._model.env.features_keys
            # Start from n_start features
            n_start = 8
            rand_tasks = self._random_task_choice(
                self._model.env.all_tasks, num_designs
            )
            self._all_designer_prior_tasks = rand_tasks
            for di, d_data in enumerate(self._design_data["DESIGNS"]):
                n_feats = di + n_start
                rand_w = self._make_random_weights(all_keys)
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                for key, val in rand_w.items():
                    if key not in true_w:
                        true_w[key] = val
                    if len((true_w.keys())) == n_feats:
                        break
                self._all_designer_prior_ws.append(true_w)

        else:
            raise NotImplementedError

    def _make_random_weights(self, keys):
        weights = OrderedDict()
        for key in keys:
            weights[key] = np.exp(self._random_weight())
        for key, val in weights.items():
            weights[key] = val / weights[self._normalized_key]
        return weights

    def run_designer(self):
        """Simulate designer on new_task. Varying the number of latent
        tasks as prior

        Args:
            num_prior_tasks (int): how many prior tasks to load
            prior_mode (str): "design" or "random"
            num_design_tasks (int): if > 1, then design a few tasks simultaneously to speed up

        """
        ## Find evaluation tasks
        # May cause high variance
        assert self._random_task_choice is not None
        all_tasks = self._designer.env.all_tasks
        viz_tasks = self._random_task_choice(all_tasks, self._num_visualize_tasks)
        viz_names = [str(t) for t in viz_tasks]

        eval_task = self._random_task_choice(all_tasks, 1)
        ## Simulate
        for n_prior in range(len(self._all_designer_prior_tasks)):

            print(f"Experiment mode {self._exp_mode}")
            print(f"Prior task number: {n_prior}")

            prior_tasks = self._all_designer_prior_tasks[:n_prior]
            if self._exp_mode == "designer_true_w_more_features":
                ## Keeps only 2 prior tasks
                prior_tasks = self._all_designer_prior_tasks[:2]
            prior_w = self._all_designer_prior_ws[n_prior]
            self._designer.prior_tasks = prior_tasks
            self._designer.true_w = prior_w
            obs = self._designer.simulate(
                eval_task, eval_task, save_name=f"designer_prior_{n_prior:02d}"
            )

            ## Visualize performance
            plot_dir = f"{self._save_dir}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            fig_name = f"prior_{n_prior:02d}"
            obs.visualize_comparisons(
                viz_tasks, viz_names, self._designer.truth, fig_name
            )

            ## Reset designer prior tasks
            self._designer.prior_tasks = all_tasks

    def run_ird(self, num_obs, mode):
        """Run IRD on task, varying the number of past observations."""
        print(f"Observation number: {num_obs}")

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

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._designer.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        random_weight = partial(
            random_uniform,
            low=-self._exp_params["MAX_WEIGHT"],
            high=self._exp_params["MAX_WEIGHT"],
        )
        self._random_weight = seed(random_weight, rng_key)
        if self._fixed_task_seed is not None:
            self._random_task_choice = seed(random_choice, self._fixed_task_seed)
        else:
            self._random_task_choice = self._random_choice
        self._load_design()
