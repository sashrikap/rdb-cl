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
        exp_params={},
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
        self._last_time = time()
        # Load design and cache
        self._design_data = design_data
        # Designer relevant data
        self._all_designer_prior_tasks = []
        self._all_designer_prior_ws = []
        self._designer = self._model.designer
        # IRD relevant data
        self._all_ird_obs_tasks = []
        self._all_ird_obs_ws = []
        self._max_ird_obs_num = 10
        # Test incremental # features
        self._nfeats_start = 5

    def _load_design(self, exp_mode):
        """Different mode of designer experiments.

        Args:
            exp_mode (str)

        """
        all_tasks = self._model.env.all_tasks
        num_designs = len(self._design_data["DESIGNS"])
        for design in self._design_data["DESIGNS"]:
            design["WEIGHTS"] = normalize_weights(
                design["WEIGHTS"], self._normalized_key
            )

        if exp_mode.startswith("ird"):
            ## Testing IRD convergence, no need of these designs
            pass

        elif "designer_convergence_true_w_prior_tasks" in exp_mode:
            ## Load well-defined Jerry's prior designs
            ## Test # design tasks vs convergence
            for d_data in self._design_data["DESIGNS"]:
                design_task = d_data["TASK"]
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_designer_prior_tasks.append(design_task)
                self._all_designer_prior_ws.append(true_w)

        elif "designer_convergence_true_w_random_tasks" in exp_mode:
            ## Use one of Jerry's prior designed w as true w
            ## Test # random tasks vs convergence
            rand_tasks = self._random_task_choice(all_tasks, num_designs)
            self._all_designer_prior_tasks = rand_tasks
            for d_data in self._design_data["DESIGNS"]:
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_designer_prior_ws.append(true_w)

        elif "designer_convergence_random_w_random_tasks" in exp_mode:
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

        elif "designer_convergence_true_w_more_features" in exp_mode:
            ## Use a true w on random tasks
            ## The true w has progressively more random features
            ## Test # features (DOF) vs convergence
            all_keys = self._model.env.features_keys
            rand_tasks = self._random_task_choice(
                self._model.env.all_tasks, num_designs
            )
            self._all_designer_prior_tasks = rand_tasks
            for di, d_data in enumerate(self._design_data["DESIGNS"]):
                # Start from n_start features
                n_feats = di + self._nfeats_start
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

    def _load_observations(self, exp_mode):
        """Different mode of IRD experiments.

        Args:
            exp_mode (str)

        """
        all_tasks = self._model.env.all_tasks
        num_obs = self._max_ird_obs_num
        for design in self._design_data["DESIGNS"]:
            design["WEIGHTS"] = normalize_weights(
                design["WEIGHTS"], self._normalized_key
            )
        rand_tasks = onp.array(self._random_task_choice(all_tasks, num_obs))

        if exp_mode.startswith("designer"):
            ## Designer experiment
            pass

        elif "ird_convergence_true_w_prior_tasks" in exp_mode:
            ## Load well-defined Jerry's prior designs
            ## Test # obs vs convergence
            for d_data in self._design_data["DESIGNS"]:
                design_task = d_data["TASK"]
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_ird_obs_tasks.append(design_task)
                self._all_ird_obs_ws.append(true_w)

        elif "ird_convergence_true_w_random_tasks" in exp_mode:
            ## Use one of Jerry's prior designed w as true w
            ## Test # random tasks vs convergence
            self._all_ird_obs_tasks = rand_tasks
            for d_data in self._design_data["DESIGNS"]:
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_ird_obs_ws.append(true_w)

        elif "ird_convergence_random_w_random_tasks" in exp_mode:
            ## Use a random w as true w
            ## Test # random tasks vs convergence
            self._all_ird_obs_tasks = rand_tasks
            rand_w = self._make_random_weights(
                self._design_data["DESIGNS"][-1]["WEIGHTS"].keys()
            )
            for d_data in self._design_data["DESIGNS"]:
                # rand_w = self._make_random_weights(d_data["WEIGHTS"].keys())
                self._all_ird_obs_ws.append(rand_w)

        elif "ird_convergence_true_w_more_features" in exp_mode:
            ## Use a true w on random tasks
            ## The true w has progressively more random features
            ## Test # features (DOF) vs convergence
            all_keys = self._model.env.features_keys
            self._all_ird_obs_tasks = rand_tasks
            for di, d_data in enumerate(self._design_data["DESIGNS"]):
                # Start from n_start features
                n_feats = di + self._nfeats_start
                rand_w = self._make_random_weights(all_keys)
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                for key, val in rand_w.items():
                    if key not in true_w:
                        true_w[key] = val
                    if len((true_w.keys())) == n_feats:
                        break
                self._all_ird_obs_ws.append(true_w)

        else:
            raise NotImplementedError

    def _make_random_weights(self, keys):
        weights = OrderedDict()
        for key in keys:
            weights[key] = np.exp(self._random_weight())
        for key, val in weights.items():
            weights[key] = val / weights[self._normalized_key]
        return weights

    def run_designer(self, exp_mode):
        """Simulate designer on new_task. Varying the number of latent
        tasks as prior

        Args:
            exp_mode (str): see `self._load_design`

        """

        self._load_design(exp_mode)
        save_dir = f"{self._save_root}/{exp_mode}"

        assert self._random_task_choice is not None
        all_tasks = self._designer.env.all_tasks
        viz_tasks = self._random_task_choice(all_tasks, self._num_visualize_tasks)
        viz_names = [str(t) for t in viz_tasks]

        new_task = self._random_task_choice(all_tasks, 1)
        ## Simulate
        for n_prior in range(len(self._all_designer_prior_tasks)):

            print(f"Experiment mode ({self._rng_key}) {exp_mode}")
            print(f"Prior task number: {n_prior}")

            prior_tasks = self._all_designer_prior_tasks[:n_prior]
            if exp_mode == "designer_convergence_true_w_more_features":
                ## Keeps only 2 prior tasks
                prior_tasks = self._all_designer_prior_tasks[:2]
            prior_w = self._all_designer_prior_ws[n_prior]
            self._designer.prior_tasks = prior_tasks
            self._designer.true_w = prior_w
            obs = self._designer.simulate(
                new_task, str(new_task), save_name=f"designer_prior_{n_prior:02d}"
            )

            ## Visualize performance
            plot_dir = f"{save_dir}/plots"
            os.makedirs(plot_dir, exist_ok=True)
            fig_name = f"prior_{n_prior:02d}"
            obs.visualize_comparisons(
                viz_tasks, viz_names, self._designer.truth, fig_name
            )

            ## Reset designer prior tasks
            self._designer.prior_tasks = all_tasks

    def run_ird(self, exp_mode):
        """Run IRD on task, varying the number of past observations."""

        self._load_observations(exp_mode)
        save_dir = f"{self._save_root}/{exp_mode}"

        assert self._random_task_choice is not None
        all_tasks = self._designer.env.all_tasks

        for num_obs in range(1, self._max_ird_obs_num):

            print(f"Experiment mode ({self._rng_key}): {exp_mode}")
            print(f"Observation number: {num_obs}")

            ## Simulate
            obs_tasks = self._all_ird_obs_tasks[:num_obs]
            obs_names = [str(task) for task in obs_tasks]
            obs_ws = self._all_ird_obs_ws[:num_obs]
            obs = [
                self._designer.create_particles(
                    [w], save_name=f"designer_ird_obs_{num_obs:02d}"
                )
                for w in obs_ws
            ]
            belief = self._model.sample(
                tasks=obs_tasks,
                task_names=obs_names,
                obs=obs,
                save_name=f"ird_obs_{num_obs:02d}",
            )

            ## Reset designer prior tasks
            self._designer.prior_tasks = all_tasks

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._designer.update_key(rng_key)
        self._model.update_key(rng_key)
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
