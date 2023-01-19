"""Test how robust IRD is to proxy.
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


class ExperimentProxyIRD(object):
    """Test how IRD is sensitive to designer proxy.
    """

    def __init__(
        self,
        model,
        env_fn,
        designer_fn,
        true_w,
        # Evaluation parameters
        num_proxies=10,
        # Initial tasks
        initial_tasks_file=None,
        # Observation model
        num_prior_tasks=0,  # for designer
        # Metadata
        save_root="data/task_beta_exp1",
        exp_name="task_beta_exp1",
        exp_params={},
    ):

        # Inverse Reward Design Model
        self._model = model
        self._env_fn = env_fn
        self._true_w = true_w

        # Random key & function
        self._rng_key = None
        self._rng_name = None
        self._num_prior_tasks = num_prior_tasks

        # Initial tasks
        self._initial_tasks_file = initial_tasks_file

        # Designer simulation
        self._designer = designer_fn()
        self._joint_mode = self._designer.design_mode == "joint"
        self._num_proxies = num_proxies

        # Save path
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        self._last_time = time.time()

    def update_key(self, rng_key):
        self._rng_name = str(rng_key)
        self._rng_key, rng_designer, rng_model, rng_choice, rng_weight = random.split(
            rng_key, 5
        )
        self._designer.update_key(rng_designer)
        self._designer.true_w = self._true_w
        self._designer.rng_name = str(rng_key)
        self._model.rng_name = str(rng_key)
        self._model.update_key(rng_model)

    def _get_rng(self, rng_type=None):
        self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()

    def run(self):
        """Main function: Run experiment."""
        print(
            f"\n============= Main Experiment ({self._rng_name}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        self._train_tasks = self._model.env.all_tasks
        prior_tasks = random_choice(
            self._get_rng("prior"),
            self._train_tasks,
            (self._num_prior_tasks,),
            replace=False,
        )
        self._designer.prior_tasks = prior_tasks

        ## Propose tasks
        tasks = onp.array(self._propose_initial_tasks())

        ## Simulate Designer
        self._log_time("Simulate Designer")
        proxies = self._designer.simulate(
            tasks=tasks, save_name=f"designer_dbeta_{self._designer.beta}"
        )
        proxies = proxies.subsample(self._num_proxies)
        self._log_time("Simulate Designer Finished")

        ## Simualte IRD (with truth)
        for ib, obs in enumerate(proxies):
            belief = self._model.sample(
                tasks=tasks,
                obs=[obs],
                save_name=f"ird_belief_truth_ibeta_{self._model.beta}_proxy_{ib:02d}",
            )
            self._log_time(f"Simulate IRD Finished {ib}/{self._num_proxies}")
        return

    def _propose_initial_tasks(self):
        filepath = f"{examples_dir()}/tasks/{self._initial_tasks_file}.yaml"
        tasks = load_params(filepath)["TASKS"]
        self._initial_tasks = tasks
        return tasks
