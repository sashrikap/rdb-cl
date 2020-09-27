"""Experiment to check the distributional performance of IRD and designer model
under different initial tasks and beta.

Observation:
    * Designer and IRD model yield different performance under different initial tasks.

Goal:
    * Quantify how initial task affects designer & IRD performance

Note:
    * Ground truth is fixed
    * Designer model takes initial tasks
    * IRD model takes GT on initial tasks

TODO:
    * Quantify how IRD works under noisy observations

"""

from rdb.distrib.designer import DesignerServer
from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler, save_params
from rdb.infer.universal import *
from numpyro.handlers import seed
from functools import partial
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
import jax.numpy as jnp
import numpy as onp
import time
import copy
import os


class ExperimentTaskBeta(object):
    """Initial task, designer and IRD beta experiment.

    """

    def __init__(
        self,
        model,
        env_fn,
        designer_fn,
        true_w,
        eval_server,
        # Evaluation parameters
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="map",
        eval_seed=None,
        universal_model=None,
        # Exp parameters
        designer_betas=[],
        ird_betas=[],
        risk_averse=False,
        # Initial tasks
        initial_tasks_file=None,
        # Observation model
        num_prior_tasks=0,  # for designer
        # Metadata
        model_dir="data/universal",
        save_root="data/task_beta_exp1",
        exp_name="task_beta_exp1",
        exp_params={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._env_fn = env_fn
        self._true_w = true_w
        self._eval_server = eval_server

        # Random key & function
        self._rng_key = None
        self._rng_name = None
        self._eval_seed = random.PRNGKey(eval_seed)
        self._num_prior_tasks = num_prior_tasks

        # Initial tasks
        self._initial_tasks_file = initial_tasks_file

        # Universal model
        self._universal_model = None
        if universal_model is not None:
            data = jnp.load(os.path.join(model_dir, universal_model), allow_pickle=True)
            _, predict = create_model(
                data["input_dim"].item(), data["output_dim"].item(), mode="test"
            )
            self._universal_model = lambda weights: predict(
                list(data["params"]), weights
            )

        # Exp parameter
        self._designer_betas = designer_betas
        self._ird_betas = ird_betas
        self._risk_averse = risk_averse

        # Designer simulation
        self._designer_server = DesignerServer(designer_fn)
        self._designer_server.register(len(designer_betas))
        self._joint_mode = self._designer_server.designer.design_mode == "joint"

        # Evaluation
        self._num_eval = num_eval
        self._eval_method = eval_method
        assert eval_method in {"map", "mean", "post_mean"}
        self._num_eval_tasks = num_eval_tasks
        self._eval_env_name = eval_env_name

        # Save path
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        self._last_time = time.time()

    def _build_cache(self):
        self._initial_tasks = None
        self._designer_proxies = {}
        self._ird_beliefs = {}
        self._ird_obs = {}
        self._designer_eval_hist = {}
        self._ird_eval_hist = {}

    def update_key(self, rng_key):
        self._rng_name = str(rng_key)
        self._rng_key, rng_designer, rng_model, rng_choice, rng_weight = random.split(
            rng_key, 5
        )
        self._designer_server.update_key(rng_designer)
        self._designer_server.set_true_w(self._true_w)
        self._designer_server.set_rng_name(str(rng_key))
        self._model.rng_name = str(rng_key)
        self._model.update_key(rng_model)

    def _get_rng(self, rng_type=None):
        if rng_type == "eval" and self._eval_seed is not None:
            self._eval_seed, rng_task = random.split(self._eval_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def _propose_initial_tasks(self):
        filepath = f"{examples_dir()}/tasks/{self._initial_tasks_file}.yaml"
        tasks = load_params(filepath)["TASKS"]
        self._initial_tasks = tasks
        return tasks

    def run(self):
        """Main function: Run experiment."""
        print(
            f"\n============= Main Experiment ({self._rng_name}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        self._build_cache()
        eval_env = self._env_fn(self._eval_env_name)
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(eval_env.all_tasks):
            num_eval = -1
        self._eval_tasks = random_choice(
            self._get_rng("eval"), eval_env.all_tasks, (num_eval,), replace=False
        )
        self._train_tasks = self._model.env.all_tasks
        prior_tasks = random_choice(
            self._get_rng("eval"),
            self._train_tasks,
            (self._num_prior_tasks,),
            replace=False,
        )
        self._designer_server.set_prior_tasks(prior_tasks)

        ## Propose tasks
        tasks = self._propose_initial_tasks()

        ## Simulate Designer
        dbetas = self._designer_betas
        if len(dbetas) > 0:
            self._designer_server.set_betas(dbetas)
            self._log_time("Simulate Designer")
            all_proxies = self._designer_server.simulate(
                onp.array([tasks] * len(dbetas)),
                methods=[f"beta_{beta}" for beta in dbetas],
                itr=0,
            )
            self._designer_proxies = dict(zip(dbetas, all_proxies))
            self._save()
            self._log_time("Simulate Designer Finished")

            ## Evaluate Designer
            for dbeta, proxies in self._designer_proxies.items():
                self._evaluate("designer", dbeta, proxies)
                self._save()
        self._log_time("Evaluate Designer Finished")

        ## Simualte IRD (with truth)
        ibetas = self._ird_betas
        if len(ibetas) > 0:
            for ibeta in ibetas:
                self._model.beta = ibeta
                obs = [self._designer_server.designer.truth] * len(tasks)
                belief = self._model.sample(
                    tasks=tasks,
                    obs=obs,
                    save_name=f"ird_belief_truth_ibeta_{ibeta}",
                    universal_model=self._universal_model,
                )
                self._ird_beliefs[ibeta] = belief
                self._ird_obs[ibeta] = obs
                self._normalize_by_obs(belief, tasks, obs)
                self._save()
            self._log_time("Simulate IRD Finished")

            ## Evaluate IRD
            for ibeta, belief in self._ird_beliefs.items():
                self._evaluate("ird", ibeta, belief)
                self._save()
        self._log_time("Evaluate IRD Finished")

        return

    def _normalize_by_obs(self, particles, tasks, norm_particles, method="first"):
        """Normalize particles based on target particles.
        Target particles are typically observations.

        Args:
            particles: Particles
            norm_particles: list(Particles) to normalize against

        """

        if method == "first":
            target_tasks = jnp.array([tasks[0]])
            targets = norm_particles[0]
            target_feats = targets.get_features(target_tasks)
            offset = particles.get_offset_by_features(target_feats)
            particles.weights = particles.weights.add_key(
                "bias", offset
            )  # nfeats * (nrisk, )
        else:
            raise NotImplementedError

    def _evaluate(self, eval_mode, beta, particles):
        """Evaluate weight particles on eval task.

        Computation: n_map(~4) * n_eval(5~10k) tasks

        Note:
            self._num_eval: number of sub samples for evaluation
            self._eval_method: use mean/mean sample

        Criteria:
            * Relative Reward.
            * Violations.

        """
        assert eval_mode in {"designer", "ird"}
        target = self._designer_server.designer.truth
        target.risk_averse = self._risk_averse

        ## Compute proxies features
        if self._eval_method == "mean":
            particles_sample = particles.subsample(self._num_eval)
        elif self._eval_method == "map":
            particles_sample = particles.map_estimate(self._num_eval, log_scale=False)

        ## Set risk averse parameter
        print(f"Eval Risk averse mode {self._risk_averse}")
        particles_sample.risk_averse = self._risk_averse

        # import pdb; pdb.set_trace()
        ## TODO: bug
        # particles_sample.compute_tasks(self._eval_tasks)
        self._eval_server.compute_tasks(
            "Evaluation", particles_sample, self._eval_tasks, verbose=True
        )
        self._eval_server.compute_tasks(
            "Evaluation", target, self._eval_tasks, verbose=True
        )
        all_violates, rel_violates = [], []
        all_performs, rel_performs = [], []
        normalized_performs = []
        feats_violates = []
        desc = f"Evaluating {eval_mode} beta {beta}"
        for task in tqdm(self._eval_tasks, desc=desc):
            comparisons = particles_sample.compare_with(task, target=target)
            import pdb

            pdb.set_trace()
            all_performs.append(comparisons["rews"].mean())
            all_violates.append(comparisons["vios"].mean())  # (nweights,) -> (1,)
            rel_performs.append(comparisons["rews_relative"].mean())
            rel_violates.append(comparisons["vios_relative"].mean())
            feats_violates.append(comparisons["vios_by_name"])
            normalized_performs.append(comparisons["rews_normalized"].mean())

        avg_violate = onp.mean(onp.array(all_violates, dtype=float))
        avg_perform = onp.mean(onp.array(all_performs, dtype=float))
        avg_rel_violate = onp.mean(onp.array(rel_violates, dtype=float))
        avg_rel_perform = onp.mean(onp.array(rel_performs, dtype=float))
        avg_feats_violate = feats_violates[0] * (1 / float(len(self._eval_tasks)))
        avg_normalized_perform = onp.mean(onp.array(normalized_performs, dtype=float))
        for fv in feats_violates[1:]:
            avg_feats_violate += fv * (1 / float(len(self._eval_tasks)))
        print(f"\t{eval_mode}({beta:.1f}) Average Violation diff {avg_violate:.4f}")
        print(f"\t{eval_mode}({beta:.1f}) Average Violation rel {avg_rel_violate:.4f}")
        print(f"\t{eval_mode}({beta:.1f}) Average Performance diff {avg_perform:.4f}")
        print(
            f"\t{eval_mode}({beta:.1f}) Average Performance normalized {avg_normalized_perform:.4f}"
        )
        print(
            f"\t{eval_mode}({beta:.1f}) Average Performance rel {avg_rel_perform:.4f}"
        )

        info = {
            "violation": avg_violate,
            "feats_violation": dict(avg_feats_violate),
            "perform": avg_perform,
            "rel_violation": avg_rel_violate,
            "rel_perform": avg_rel_perform,
            "normalized_perform": avg_normalized_perform,
        }
        if eval_mode == "designer":
            self._designer_eval_hist[beta] = info
        elif eval_mode == "ird":
            self._ird_eval_hist[beta] = info
        else:
            raise NotImplementedError
        return info

    def _save(self, skip_weights=False):
        """Save checkpoint.

        Format:
            * data/save_root/exp_name/{exp_name}_seed.npz
            * data/save_root/exp_name/save/weights_seed_method_itr.npz
              - see `rdb.infer.particles.save()`

        """
        print("Saving to:", self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        ## Save experiment parameters
        save_params(f"{self._save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        data = dict(
            seed=self._rng_name,
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            true_w=self._designer_server.designer.true_w,
            initial_tasks=self._initial_tasks,
            designer_eval_hist=self._designer_eval_hist,
            ird_eval_hist=self._ird_eval_hist,
            eval_tasks=self._eval_tasks
            if self._num_eval_tasks > 0
            else [],  # do not save when eval onall tasks (too large)
        )
        npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        with open(npz_path, "wb+") as f:
            jnp.savez(f, **data)
        ## Save evaluation history to yaml
        npy_path = (
            f"{self._save_dir}/{self._exp_name}_designer_seed_{self._rng_name}.npy"
        )
        jnp.save(npy_path, self._designer_eval_hist)
        npy_path = f"{self._save_dir}/{self._exp_name}_ird_seed_{self._rng_name}.npy"
        jnp.save(npy_path, self._ird_eval_hist)

        if not skip_weights:
            for dbeta, proxies in self._designer_proxies.items():
                proxies.save()
            for ibeta, belief in self._ird_beliefs.items():
                belief.save()
        print("Saving done")

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()
