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
from rdb.exps.utils import Profiler, save_params
# from numpyro.handlers import seed
from functools import partial
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
import jax.numpy as jnp
import numpy as onp
import time
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
        self._normalized_key = normalized_key
        self._fixed_task_seed = fixed_task_seed
        # Evaluation & Visualization
        self._num_eval_map = num_eval_map
        self._num_eval_tasks = num_eval_tasks
        self._num_visualize_tasks = num_visualize_tasks
        # Active Task proposal
        self._exp_params = exp_params
        self._save_root = save_root
        self._last_time = time.time()
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
        self._last_time = None

    def _get_rng_task(self):
        if self._fixed_task_seed is not None:
            self._fixed_task_seed, rng_task = random.split(self._fixed_task_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

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
                true_w = {
                    "control": 3.0,
                    "dist_cars": 0.001,
                    "dist_fences": 0.001,
                    "dist_lanes": 1.0,
                    "dist_objects": 0.001,
                    "speed": 1.0,
                    "speed_over": 0.001,
                }
                self._all_designer_prior_tasks.append(design_task)
                self._all_designer_prior_ws.append(true_w)

        elif "designer_convergence_true_w_random_tasks" in exp_mode:
            ## Use one of Jerry's prior designed w as true w
            ## Test # random tasks vs convergence
            rand_tasks = random_choice(self._get_rng_task(), all_tasks, (num_designs,))
            self._all_designer_prior_tasks = rand_tasks
            for d_data in self._design_data["DESIGNS"]:
                # Treat final design as true_w
                true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
                self._all_designer_prior_ws.append(true_w)

        elif "designer_convergence_random_w_random_tasks" in exp_mode:
            ## Use a random w as true w
            ## Test # random tasks vs convergence
            rand_tasks = onp.array(
                random_choice(self._get_rng_task(), all_tasks, (num_designs,))
            )
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
            rand_tasks = random_choice(
                self._get_rng_task(), self._model.env.all_tasks, (num_designs,)
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
        num_tasks = 100
        for design in self._design_data["DESIGNS"]:
            design["WEIGHTS"] = normalize_weights(
                design["WEIGHTS"], self._normalized_key
            )
        rand_tasks = onp.array(
            random_choice(self._get_rng_task(), all_tasks, (num_tasks,))
        )

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
            self._all_ird_obs_ws = [true_w] * num_tasks

        elif "ird_convergence_true_w_random_tasks" in exp_mode:
            ## Use one of Jerry's prior designed w as true w
            ## Test # random tasks vs convergence
            self._all_ird_obs_tasks = rand_tasks
            true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
            self._all_ird_obs_ws = [true_w] * num_tasks

        elif "ird_convergence_true_w_same_tasks" in exp_mode:
            ## Use one of Jerry's prior designed w as true w
            ## Test # same tasks vs convergence
            self._all_ird_obs_tasks = onp.array([rand_tasks[0]] * num_tasks)
            true_w = self._design_data["DESIGNS"][-1]["WEIGHTS"]
            self._all_ird_obs_ws = [true_w] * num_tasks

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
            self._all_ird_obs_ws = [true_w] * num_tasks

        else:
            raise NotImplementedError

    def _make_random_weights(self, keys):
        weights = OrderedDict()
        for key in keys:
            if key == self._normalized_key:
                weights[key] = 1.0
            else:
                max_val = self._exp_params["WEIGHT_PARAMS"]["max_weights"]
                self._rng_key, rng_weights = random.split(self._rng_key)
                weights[key] = jnp.exp(
                    random.uniform(rng_weights, minval=-max_val, maxval=max_val)
                )
        return weights

    def run_designer(self, exp_mode):
        """Simulate designer on new_task. Varying the number of latent
        tasks as prior

        Args:
            exp_mode (str): see `self._load_design`

        """

        self._load_design(exp_mode)
        save_dir = f"{self._save_root}/{exp_mode}"
        save_params(f"{save_dir}/params_{self._rng_name}.yaml", self._exp_params)

        all_tasks = self._designer.env.all_tasks
        viz_tasks = random_choice(
            self._get_rng_task(), all_tasks, (self._num_visualize_tasks,)
        )

        # new_tasks = random_choice(self._get_rng_task(), all_tasks, 1)
        filepath = f"{examples_dir()}/tasks/Week6_04_1v2.yaml"
        initial_task = load_params(filepath)["TASKS"][0]
        new_tasks = onp.array([initial_task])
        ## Simulate
        # for n_prior in range(len(self._all_designer_prior_tasks)):
        for n_prior in range(0, 6):

            self._log_time(f"Designer Prior {n_prior} Begin")
            print(f"Experiment mode ({self._rng_name}) {exp_mode}")
            print(f"Prior task number: {n_prior}")

            prior_tasks = self._all_designer_prior_tasks[:n_prior]
            if "designer_convergence_true_w_more_features" in exp_mode:
                ## Keeps only 2 prior tasks
                prior_tasks = self._all_designer_prior_tasks[:2]
            prior_w = self._all_designer_prior_ws[n_prior]

            ## Set designer data
            # import pdb; pdb.set_trace()
            self._designer.prior_tasks = prior_tasks
            self._designer.true_w = prior_w

            ## Sample Designer
            num_samples = self._exp_params["DESIGNER_ARGS"]["sample_args"][
                "num_samples"
            ]
            save_name = f"designer_sample_{num_samples:04d}_prior_{n_prior:02d}"
            samples = self._designer.simulate(new_tasks, save_name=save_name)
            import pdb

            pdb.set_trace()
            ## Visualize performance
            # obs.visualize_comparisons(
            #     tasks=viz_tasks,
            #     target=self._designer.truth,
            #     fig_name=f"prior_{n_prior:02d}",
            # )

            ## Reset designer prior tasks
            self._designer.prior_tasks = all_tasks
            self._log_time(f"Designer Prior {n_prior} End")

    def run_ird(self, exp_mode):
        """Run IRD on task, varying the number of past observations."""

        self._load_observations(exp_mode)
        save_dir = f"{self._save_root}/{exp_mode}"
        save_params(f"{save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        all_tasks = self._designer.env.all_tasks

        # for num_obs in range(6, len(self._all_ird_obs_ws)):
        # for num_obs in range(1, 6):
        for num_obs in range(1, 15):
            # for num_obs in range(4, 5):
            # for num_obs in range(3, 6):

            self._log_time(f"IRD Obs {num_obs} Begin")
            print(f"Experiment mode ({self._rng_name}): {exp_mode}")
            print(f"Observation number: {num_obs}")

            ## Simulate
            obs_tasks = self._all_ird_obs_tasks[:num_obs]
            obs_ws = self._all_ird_obs_ws[:num_obs]
            obs_name = f"designer_ird_obs_{num_obs:02d}"
            obs = [
                self._designer.create_particles([w], save_name=obs_name) for w in obs_ws
            ]
            num_samples = self._exp_params["IRD_ARGS"]["sample_args"]["num_samples"]
            belief_name = f"ird_sample_{num_samples:04d}_obs_{num_obs:02d}"
            belief = self._model.sample(tasks=obs_tasks, obs=obs, save_name=belief_name)
            belief.save()
            # import pdb; pdb.set_trace()

            ## Reset designer prior tasks
            self._designer.prior_tasks = all_tasks
            self._log_time(f"IRD Obs {num_obs} End")

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()

    def update_key(self, rng_key):
        self._rng_name = str(rng_key)
        self._rng_key, rng_designer, rng_model, rng_choice, rng_weight = random.split(
            rng_key, 5
        )

        self._designer.rng_name = str(rng_key)
        self._designer.update_key(rng_designer)
        self._model.rng_name = str(rng_key)
        self._model.update_key(rng_model)
