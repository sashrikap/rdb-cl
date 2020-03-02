"""Active-IRD Main Experiment.

Prerequisite:
    * Acquisition criteria
    * Environment (rdb.envs.drive2d)
    * Belief sampler (rdb.infer.ird_oc.py)

Full Active-IRD Evaluation:
    1) Choose True weight, environment
    2) Simulate Designer on env
    3) Infer b(w) over true weight, sample environment
    4) Evaluate current MAP w
    5) Go back to step 2)

How to tune hyperparameters in Active IRD (By Jerry)
    1) IRD_PROPOSAL_VAR and DESIGNER_VAR are most sensitive for sampling
    2) Tune BETA with `rdb/examples/run_dsigner.py`
    3) Tun NUM_WARMUPS/NUM_SAMPLES with different combinations
    4) Control for randomness with `rdb/examples/run_active.py`

How to control randomness (By Jerry)
    1) In general, run with >=4 different seeds
    2) Evaluation: # MAP, # Tasks
    3) Active: # Candidates, # Histogram bins
    4) MH Sampling: # Designer, # IRD Samples, # Warmup

Note:
    * see rdb/exps/active.py for acquisition functions

Credits:
    * Jerry Z. He 2019-2020

"""

from rdb.infer.utils import random_choice
from rdb.infer.particles import Particles
from numpyro.handlers import seed
from rdb.visualize.plot import *
from rdb.exps.utils import *
from tqdm.auto import tqdm
from jax import random
from time import time
import jax.numpy as np
import numpy as onp
import copy
import yaml
import os


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        active_fns (dict): map name -> active function
        model (object): IRD model
        iteration (int): algorithm iterations
        num_eval_map (int): if > 0, use MAP estimate
        num_active_tasks (int): # task candidates for active selection
        num_active_sample (int): running acquisition function on belief
            samples is costly, so subsample belief particles

    """

    def __init__(
        self,
        model,
        designer,
        active_fns,
        true_w,
        eval_server,
        iterations=10,
        num_eval_tasks=4,
        num_eval_map=-1,
        num_active_tasks=4,
        num_active_sample=-1,
        fixed_task_seed=None,
        fixed_candidates=None,
        fixed_belief_tasks=None,
        normalized_key=None,
        design_data=None,
        num_load_design=-1,
        save_root="data/active_ird_exp1",
        exp_name="active_ird_exp1",
        exp_params={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._designer = designer
        self._true_w = true_w
        self._active_fns = active_fns
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        self._rng_name = None
        self._fixed_task_seed = fixed_task_seed
        self._iterations = iterations
        self._normalized_key = normalized_key
        # Evaluation
        self._num_eval_map = num_eval_map
        self._num_eval_tasks = num_eval_tasks
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._fixed_candidates = fixed_candidates
        self._num_active_sample = num_active_sample
        self._fixed_belief_tasks = fixed_belief_tasks
        # Save path
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        self._last_time = time()
        # Load design and cache
        self._num_load_design = num_load_design
        self._design_data = design_data

    def _build_cache(self):
        """ Build cache data """
        self._initial_task = None
        self._active_eval_hist = {}
        self._active_map_vs_obs_hist = {}
        self._all_tasks = {}
        self._all_obs, self._all_beliefs = {}, {}
        self._all_candidates = []
        self._all_cand_scores = {}
        self._eval_tasks = []
        self._curr_belief = {}
        for key in self._active_fns.keys():
            self._active_eval_hist[key] = []
            self._active_map_vs_obs_hist[key] = []
            self._all_cand_scores[key] = []
            self._all_tasks[key] = []
            self._all_obs[key], self._all_beliefs[key] = [], []

    def update_key(self, rng_key):
        # Set name
        self._rng_name = str(rng_key)
        self._model.rng_name = str(rng_key)
        self._designer.rng_name = str(rng_key)
        # Model and designer
        self._rng_key, rng_model, rng_designer, rng_choice, rng_active = random.split(
            rng_key, 5
        )
        self._model.update_key(rng_model)
        self._designer.update_key(rng_designer)
        self._designer.true_w = self._true_w
        # Active functions
        rng_active_keys = random.split(rng_active, len(list(self._active_fns.keys())))
        for fn, rng_key_fn in zip(list(self._active_fns.values()), rng_active_keys):
            fn.update_key(rng_key_fn)

    def _get_rng_task(self):
        if self._fixed_task_seed is not None:
            self._fixed_task_seed, rng_task = random.split(self._fixed_task_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def run(self, plot_candidates=False):
        """Main function 1: Run experiment from task.

        Args:
            # task (obj): initial task; if None, start from random.
            plot_candidates (bool): plot rankings of proposed candidates

        """
        print(
            f"\n============= Main Experiment ({self._rng_name}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        self._build_cache()
        self._load_design(self._num_load_design, self._design_data)
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(self._model.env.all_tasks):
            num_eval = -1
        self._eval_tasks = random_choice(
            self._get_rng_task(), self._model.env.all_tasks, num_eval
        )

        ### Main Experiment Loop ###
        for itr in range(0, self._iterations):
            ## Candidates for next tasks
            candidates = random_choice(
                self._get_rng_task(), self._model.env.all_tasks, self._num_active_tasks
            )
            self._all_candidates.append(candidates)

            ### Run Active IRD on Candidates ###
            print(f"\nActive IRD ({self._rng_name}) iteration {itr}")
            for key in self._active_fns.keys():
                all_beliefs = self._all_beliefs[key]
                all_obs = self._all_obs[key]
                all_tasks = self._all_tasks[key]
                print(f"Method: {key}")

                ## Actively Task Proposal
                if len(all_obs) == 0:
                    task, scores = self._propose_random_task()
                else:
                    task, scores = self._propose_task(
                        candidates, all_beliefs, all_obs, all_tasks, key
                    )

                self._all_tasks[key].append(task)
                self._all_cand_scores[key].append(scores)
                self._plot_candidate_scores(itr)
                self._log_time(f"Itr {itr} {key} Propose")
                if len(all_beliefs) > 0:
                    self._compare_on_next(key, all_beliefs[-1], all_obs[-1], task)

                ## Simulate Designer
                obs = self._designer.simulate(
                    onp.array([task]), save_name=f"designer_method_{key}_itr_{itr:02d}"
                )
                self._all_obs[key].append(obs)
                self._log_time(f"Itr {itr} {key} Designer")

                ## IRD Sampling w/ Divide And Conquer
                belief = self._model.sample(
                    tasks=onp.array(self._all_tasks[key]),
                    obs=self._all_obs[key],
                    save_name=f"ird_belief_method_{key}_itr_{itr:02d}",
                )
                self._all_beliefs[key].append(belief)
                self._curr_belief[key] = belief
                self._log_time(f"Itr {itr} {key} IRD Divide & Conquer")

                ## Evaluate, plot and Save
                self._evaluate(key, belief, self._eval_tasks)
                self._save(itr=itr)
                self._log_time(f"Itr {itr} {key} Eval & Save")

    def _propose_random_task(self):
        """Used to choose first task, when user initial design is not provided.
        """

        assert (
            self._num_load_design < 0
        ), "Should not propose random task if user design is provided"
        if self._initial_task is None:
            task = random_choice(self._get_rng_task(), self._model.env.all_tasks, 1)[0]
            self._initial_task = task
        else:
            task = self._initial_task
        scores = onp.zeros(self._num_active_tasks)
        return task, scores

    def _propose_task(self, candidates, all_beliefs, all_obs, all_tasks, fn_key):
        """Find best next task for this active function.

        Computation: n_particles(~1k) * n_active(~100) tasks

        Args:
            candidates (list): potential next tasks
            all_beliefs (list): all beliefs so far, usually only the last one is useful
            all_obs (Particles.weights[1]): all observations so far
            all_tasks (list): all tasks proposed so far
            fn_key (str): acquisition function key

        Note:
            * Require `tasks = env.sample_task()`
            * Use small task space to avoid being too slow.

        """

        assert (
            len(all_beliefs) == len(all_obs) == len(all_tasks)
        ), "Observation and tasks mismatch"
        assert len(all_obs) > 0, "Need at least 1 observation"
        # Compute belief features
        belief = all_beliefs[-1]
        belief = belief.subsample(self._num_active_sample)
        active_fn = self._active_fns[fn_key]
        print(f"Active proposal method {fn_key}: Begin")
        if fn_key != "random":
            ## =============== Pre-empt heavy computations =====================
            self._eval_server.compute_tasks("Active", belief, candidates, verbose=True)
            self._eval_server.compute_tasks(
                "Active", all_obs[-1], candidates, verbose=True
            )

        scores = []
        desc = "Evaluaitng candidate tasks"
        feats_keys = self._model._env.features_keys
        for next_task in tqdm(candidates, desc=desc):
            scores.append(
                active_fn(
                    onp.array([next_task]), belief, all_obs, feats_keys, verbose=False
                )
            )
        scores = onp.array(scores)

        print(f"Function {fn_key} chose task {onp.argmax(scores)} among {len(scores)}")
        next_task = candidates[onp.argmax(scores)]
        return next_task, scores

    def _evaluate(self, fn_key, belief, eval_tasks, map_eval=None, cache=True):
        """Evaluate current sampled belief on eval task.

        Computation: n_map(~4) * n_eval(5~10k) tasks

        Note:
            self._num_eval_map: use MAP estimate particle, intead of whole population, to estimate.

        Criteria:
            * Relative Reward.
            * Violations.

        """
        if map_eval is None:
            map_eval = self._num_eval_map
        if self._model.interactive_mode and self._designer.run_from_ipython():
            # Interactive mode, skip evaluation to speed up
            avg_violate = 0.0
            avg_perform = 0.0
            log_prob_true = 0.0
            feats_violate = {}
        else:
            # Compute belief features
            belief_map = belief.map_estimate(map_eval, log_scale=False)
            target = self._designer.truth
            print(f"Evaluating method {fn_key}: Begin")
            self._eval_server.compute_tasks(
                "Evaluation", belief_map, eval_tasks, verbose=True
            )
            self._eval_server.compute_tasks(
                "Evaluation", target, eval_tasks, verbose=True
            )

            num_violate = 0.0
            performance = 0.0
            feats_violate = None
            desc = f"Evaluating method {fn_key}"
            for task in tqdm(eval_tasks, desc=desc):
                diff_perf, diff_vios_arr, diff_vios = belief_map.compare_with(
                    task, target=target
                )
                performance += diff_perf.mean()
                num_violate += diff_vios_arr.mean()  # (nweights,) -> (1,)
                if feats_violate is None:
                    feats_violate = diff_vios
                else:
                    feats_violate += diff_vios
                # import pdb; pdb.set_trace()
            avg_violate = num_violate * (1 / float(len(eval_tasks)))
            avg_perform = performance * (1 / float(len(eval_tasks)))
            feats_violate = feats_violate * (1 / float(len(eval_tasks)))
            if self._designer.truth is not None:
                log_prob_true = float(belief.log_prob(self._designer.truth.weights[0]))
            else:
                log_prob_true = 0.0
            print(f"    Average Violation diff {avg_violate:.2f}")
            print(f"    Average Performance diff {avg_perform:.2f}")
            print(f"    True Weights Log Prob {log_prob_true:.2f}")

            # For comparison debugging
            # self._divide_violations = belief_map.get_violations(eval_tasks)
            # self._joint_violations = belief_map.get_violations(eval_tasks)
            # self._divide_one = self._divide_violations[:, 0]
            # self._joint_one = self._joint_violations[:, 0]
            # import pdb; pdb.set_trace()
        info = {
            "violation": avg_violate,
            "feats_violation": dict(feats_violate),
            "perform": avg_perform,
            "log_prob_true": log_prob_true,
        }
        if cache:
            self._active_eval_hist[fn_key].append(info)
        return info

    def _compare_on_next(self, fn_key, belief, joint_obs, next_task, cache=True):
        """Compare observation and MAP estimate on next proposed task"""
        belief_map = belief.map_estimate(self._num_eval_map, log_scale=False)
        print(f"Evaluating method {fn_key}: Begin")
        belief_map.compute_tasks([next_task])
        joint_obs.compute_tasks([next_task])

        target = self._designer.truth
        desc = f"Evaluating method {fn_key}"
        diff_perf, diff_vios_arr, diff_vios = belief_map.compare_with(
            next_task, target=target
        )
        map_perform = list(diff_perf)
        map_violate = list(diff_vios_arr)  # (nweights,)
        diff_perf, diff_vios_arr, diff_vios = joint_obs.compare_with(
            next_task, target=target
        )
        obs_perform = diff_perf.mean()
        obs_violate = diff_vios_arr.mean()

        print(f"    MAP Violation diff {map_violate:.2f}")
        print(f"    MAP Performance diff {map_perform:.2f}")
        print(f"    Obs Violation diff {obs_violate:.2f}")
        print(f"    Obs Performance diff {obs_perform:.2f}")
        info = {
            "map_violation": map_violate,
            "obs_violation": obs_violate,
            "map_perform": map_perform,
            "obs_perform": obs_perform,
        }
        if cache:
            self._active_map_vs_obs_hist[fn_key].append(info)

    def _load_design(self, num_load, design_data, cache=True, compute_belief=True):
        """Load previous weight designs before experiment.

        In real applications, usually designers begin with a few (e.g. 5) environments
        they have in mind, and a rough design based on them. We load these designs and
        perform active ird from there.

        Note:
            design_data (dict):
            {
                'ENV_NAME': gym environment name,
                'DESIGNS':
                [ # list of environment & design pairs
                    {
                        TASK: ...,
                        WEIGHTS: {...}
                    },
                ]
            }
            num_load (int): how many previous designs do we use

        """
        if num_load <= 0:
            return
        assert (
            design_data["ENV_NAME"] == self._exp_params["ENV_NAME"]
        ), "Environment name mismatch"
        assert len(design_data["DESIGNS"]) >= num_load, "Not enough designs to load"

        load_designs = design_data["DESIGNS"][:num_load]
        load_obs, load_tasks = [], []
        for i, design_i in enumerate(load_designs):
            # Load previous design
            task = design_i["TASK"]
            weights = normalize_weights(design_i["WEIGHTS"], self._normalized_key)
            obs = self._model.create_particles(
                [weights],
                save_name=f"ird_prior_design_{i:02d}",
                controller=self._designer._sample_controller,
                runner=self._designer._sample_runner,
            )
            load_obs.append(obs)
            load_tasks.append(task)

            if cache:
                # Cache previous designs for active functions
                for key in self._active_fns.keys():
                    # Make sure nothing cached before design data
                    assert len(self._all_obs[key]) == i
                    assert len(self._all_tasks[key]) == i
                    self._all_obs[key].append(obs)
                    self._all_tasks[key].append(task)

        print(f"Loaded {len(load_designs)} prior designs.")
        ## Compute IRD Belief based on loaded data
        if compute_belief:
            belief = self._model.sample(
                self._all_tasks[key],
                obs=self._all_obs[key],
                save_name=f"ird_prior_belief_{num_load:02d}",
            )
            for i in range(num_load):
                # Pack the same beliefs into belief history
                assert len(self._all_beliefs[key]) == i
                self._all_beliefs[key].append(belief)
        return load_obs, load_tasks

    def _load_cache(self, load_dir, load_eval=True):
        """Load previous experiment checkpoint.

        The opposite ove self._save().

        """
        # Load eval data
        self._build_cache()
        eval_path = (
            f"{load_dir}/{self._exp_name}/{self._exp_name}_seed_{self._rng_name}.npz"
        )
        if not os.path.isfile(eval_path):
            print(f"Failed to load {eval_path}")
            return False

        eval_data = np.load(eval_path, allow_pickle=True)
        if load_eval:
            self._active_eval_hist = eval_data["eval_hist"].item()
            self._active_map_vs_obs_hist = eval_data["active_map_vs_obs_hist"].item()
        self._all_candidates = eval_data["candidate_tasks"]
        self._all_cand_scores = eval_data["candidate_scores"].item()
        self._eval_tasks = eval_data["eval_tasks"]
        if len(self._eval_tasks) == 0:
            self._eval_tasks = self._model.env.all_tasks
        self._all_tasks = eval_data["curr_tasks"].item()

        # Load observations
        np_obs = eval_data["curr_obs"].item()
        for key in np_obs.keys():
            self._all_obs[key] = [
                self._model.create_particles(
                    [ws],
                    controller=self._designer._sample_controller,
                    runner=self._designer._sample_runner,
                    save_name="observation",
                )
                for ws in np_obs[key]
            ]
        if "env_id" in eval_data:
            assert self._model.env_id == eval_data["env_id"].item()
        # Load truth
        if "true_w" in eval_data:
            true_ws = [eval_data["true_w"].item()]
            self._designer.truth.weights = true_ws
        # Load parameters and check
        if "exp_params" in eval_data:
            exp_params = eval_data["exp_params"].item()
            for key, val in exp_params.items():
                assert (
                    key in self._exp_params and self._exp_params[key] == val
                ), f"Parameter changed {key}: {val}"

        # Load beliefs
        weight_dir = f"{load_dir}/{self._exp_name}/save"
        for key in self._active_fns.keys():
            weight_files = sorted(
                [f for f in os.listdir(weight_dir) if key in f and self._rng_name in f]
            )
            for file in weight_files:
                # file: weights_seed_[ 0 10]_ird_belief_method_infogain_itr_00.npz
                save_name = file[file.index("ird") :].replace(".npz", "")
                belief = self._model.create_particles(
                    weights=None,
                    controller=self._model._sample_controller,
                    runner=self._model._sample_runner,
                    save_name=save_name,
                )
                belief.load()
                self._all_beliefs[key].append(belief)
        # Load random seed
        rng_key = str_to_key(eval_data["seed"].item())
        self.update_key(rng_key)
        return True

    def _save(self, itr, fn_key=None, skip_weights=False):
        """Save checkpoint.

        Format:
            * data/save_root/exp_name/{exp_name}_seed.npz
              - seed (str): rng_key
              - curr_obs (dict): {method: [obs_w] * num_itr}
              - curr_tasks (dict): {method: [task] * num_eval}
              - eval_tasks (dict): {method: [task] * num_eval}
              - eval_hist (dict): {method: [
                    {"violation": ..., "perform": ...}
                )] * num_itr}
            * data/save_root/exp_name/save/weights_seed_method_itr.npz
              - see `rdb.infer.particles.save()`

        """
        print("Saving to:", self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        ## Save experiment parameters
        save_params(f"{self._save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        ## Save experiment history to npz
        np_obs = {}
        for key in self._all_obs.keys():
            np_obs[key] = [dict(ob.weights[0]) for ob in self._all_obs[key]]
        data = dict(
            seed=self._rng_name,
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            true_w=self._designer.true_w,
            curr_obs=np_obs,
            curr_tasks=self._all_tasks,
            eval_hist=self._active_eval_hist,
            active_map_vs_obs_hist=self._active_map_vs_obs_hist,
            candidate_tasks=self._all_candidates,
            candidate_scores=self._all_cand_scores,
            eval_tasks=self._eval_tasks
            if self._num_eval_tasks > 0
            else [],  # do not save when eval onall tasks (too large)
        )
        npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        with open(npz_path, "wb+") as f:
            np.savez(f, **data)
        ## Save evaluation history to yaml
        yaml_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.yaml"
        with open(yaml_path, "w+") as stream:
            yaml.dump(self._active_eval_hist, stream, default_flow_style=False)
        ## Save active comparison to yaml
        npy_path = f"{self._save_dir}/{self._exp_name}_map_seed_{self._rng_name}.npy"
        np.save(npy_path, self._active_map_vs_obs_hist)

        if fn_key is not None:
            ## Save belief sample information
            true_w = self._designer.true_w
            # Ony save last belief, to save time
            itr = len(self._all_beliefs[fn_key]) - 1
            belief = self._all_beliefs[fn_key][-1]
            if itr < len(np_obs[fn_key]):
                obs_w = np_obs[fn_key][itr]
            else:
                obs_w = None
            if not skip_weights:
                belief.save()
        print("Saving done")

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time()

    def run_inspection(self, override=False):
        """For interactive mode. Since it's costly to run evaluation during interactive
        mode, we save the beliefs and evaluate post-hoc.

        Args:
            override (bool): if false, do not re-calculate evaluations.

        """
        print(
            f"\n============= Evaluate Candidates ({self._rng_name}): {self._save_root} {self._exp_name} ============="
        )
        if not self._load_cache(self._save_root):
            return

        ## IRD Sampling w/ Divide And Conquer
        # key = "infogain"
        # itr = 2
        # map_weights = self._all_beliefs[key][itr].map_estimate(4).weights
        # belief = self._model.sample(tasks=onp.array(self._all_tasks[key])[:itr+1],obs=self._all_obs[key][:itr+1],save_name=f"ird_belief_method_{key}_itr_{itr:02d}",)

        all_cands = self._all_candidates
        all_scores = self._all_cand_scores
        all_beliefs = self._all_beliefs
        eval_tasks = self._eval_tasks
        num_iter = all_cands.shape[0]
        print(f"Methods {list(all_beliefs.keys())}, iterations {num_iter}")
        for itr in range(num_iter):
            for key in all_beliefs:
                if len(self._all_tasks[key]) <= itr:
                    continue
                import pdb

                pdb.set_trace()
                #  shape (1, task_dim)
                curr_task = self._all_tasks[key][itr][None, :]
                curr_obs = self._all_obs[key][itr]
                #  shape (1, task_dim)
                next_task = self._all_tasks[key][itr + 1][None, :]

                self._eval_server.compute_tasks(
                    "Evaluation", curr_obs, eval_tasks, verbose=True
                )
                self._eval_server.compute_tasks(
                    "Evaluation", curr_obs, next_task, verbose=True
                )
                target = self._designer.truth
                self._eval_server.compute_tasks(
                    "Evaluation", target, eval_tasks, verbose=True
                )
                self._eval_server.compute_tasks(
                    "Evaluation", target, next_task, verbose=True
                )

                eval_violate = 0.0
                eval_perform = 0.0
                desc = f"Evaluating method {fn_key}"
                for task in tqdm(eval_tasks, desc=desc):
                    diff_perf, diff_vios_arr, _ = curr_obs.compare_with(
                        task, target=target
                    )
                    eval_perform += diff_perf.mean() / len(eval_tasks)
                    eval_violate += diff_vios_arr.mean() / len(eval_tasks)

                cand_violate = 0.0
                cand_perform = 0.0
                desc = f"Evaluating method {fn_key}"
                for task in tqdm(eval_tasks, desc=desc):
                    diff_perf, diff_vios_arr, _ = curr_obs.compare_with(
                        task, target=target
                    )
                    cand_perform += diff_perf.mean()
                    cand_violate += diff_vios_arr.mean()

                self._save(itr=itr)
                self._log_time(f"Itr {itr} Method {key} Eval & Save")

    def run_evaluation(self):
        """For interactive mode. Since it's costly to run evaluation during interactive
        mode, we save the beliefs and evaluate post-hoc.

        Args:
            override (bool): if false, do not re-calculate evaluations.

        """
        print(
            f"\n============= Evaluate Candidates ({self._rng_name}): {self._save_root} {self._exp_name} ============="
        )
        if not self._load_cache(self._save_root):
            return

        ## IRD Sampling w/ Divide And Conquer
        key = "infogain"
        itr = 2
        map_weights = self._all_beliefs[key][itr].map_estimate(4).weights
        belief = self._model.sample(
            tasks=onp.array(self._all_tasks[key])[: itr + 1],
            obs=self._all_obs[key][: itr + 1],
            save_name=f"ird_belief_method_{key}_itr_{itr:02d}",
        )

        all_cands = self._all_candidates
        all_scores = self._all_cand_scores
        all_beliefs = self._all_beliefs
        eval_tasks = self._eval_tasks
        num_iter = all_cands.shape[0]
        print(f"Methods {list(all_beliefs.keys())}, iterations {num_iter}")
        for itr in range(num_iter):
            for key in all_beliefs:
                if len(all_beliefs[key]) <= itr:
                    continue
                belief = all_beliefs[key][itr]

    def run_comparison(self, num_tasks, design):
        """
        2020 Feb 25th: compare batch ird vs divide/conquer ird vs final design.

        """
        print(
            f"\n============= Evaluate Candidates ({self._rng_name}): {self._save_root} {self._exp_name} ============="
        )
        # assert self._load_cache(self._save_root), "Saved data not successfully loaded"
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(self._model.env.all_tasks):
            num_eval = -1
        self._eval_tasks = random_choice(
            self._get_rng_task(), self._model.env.all_tasks, num_eval
        )
        ENV_NAME = self._exp_params["ENV_NAME"]
        joint_data = load_params(
            join(
                examples_dir(),
                f"designs/{ENV_NAME}_joint_{design:02d}_num_{num_tasks:02d}.yaml",
            )
        )
        divid_data = load_params(
            join(
                examples_dir(),
                f"designs/{ENV_NAME}_divide_{design:02d}_num_{num_tasks:02d}.yaml",
            )
        )
        joint_obs, joint_tasks = self._load_design(
            num_tasks, joint_data, cache=False, compute_belief=False
        )
        divid_obs, divid_tasks = self._load_design(
            num_tasks, divid_data, cache=False, compute_belief=False
        )
        ## Do inference on all design environments with IRD & divide/conquer
        divid_belief = self._model.sample(
            tasks=onp.array(divid_tasks),
            obs=divid_obs,
            save_name=f"ird_belief_divid_tasks_{num_tasks:02d}",
        )
        divid_belief.save()
        divid_info = self._evaluate(
            "Divide", divid_belief, self._eval_tasks, cache=False
        )
        divid_vio = divid_info["violation"]

        ## Do inference on all design environments with IRD & joint design
        joint_belief = self._model.sample(
            tasks=onp.array(joint_tasks),
            obs=[joint_obs[-1]] * len(joint_tasks),
            save_name=f"ird_belief_joint_tasks_{num_tasks:02d}",
        )
        joint_belief.save()
        joint_info = self._evaluate(
            "Joint", joint_belief, self._eval_tasks, cache=False
        )
        joint_vio = joint_info["violation"]

        ## Evaluate joint IRD, d/c IRD, joint design
        designer_info = self._evaluate(
            "Designer", joint_obs[-1], self._eval_tasks, map_eval=1, cache=False
        )
        designer_vio = designer_info["violation"]
        comparison_data = {
            "joint": joint_vio,
            "joint_feats": joint_info["feats_violation"],
            "divide": divid_vio,
            "divide_feats": divid_info["feats_violation"],
            "designer": designer_vio,
            "designer_feats": designer_info["feats_violation"],
        }
        npy_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npy"
        onp.save(npy_path, comparison_data)

    def _plot_candidate_scores(self, itr, debug_dir=None):
        if debug_dir is None:
            debug_dir = self._save_root
        all_scores = self._all_cand_scores
        ranking_dir = os.path.join(debug_dir, self._exp_name, "candidates")
        os.makedirs(ranking_dir, exist_ok=True)
        for fn_key in all_scores.keys():
            # Check current active function
            if len(all_scores[fn_key]) <= itr:
                continue
            cand_scores = onp.array(all_scores[fn_key][itr])
            other_scores = []
            other_keys = []
            # Check other active function
            other_scores_all = copy.deepcopy(all_scores)
            del other_scores_all[fn_key]
            for key in other_scores_all.keys():
                if len(other_scores_all[key]) > itr:
                    other_scores.append(other_scores_all[key][itr])
                    other_keys.append(key)

            # Ranking plot
            file = f"key_{self._rng_name}_cand_itr_{itr}_fn_{fn_key}_ranking.png"
            path = os.path.join(ranking_dir, file)
            print(f"Candidate plot saved to {path}")
            plot_rankings(
                cand_scores,
                fn_key,
                other_scores,
                other_keys,
                path=path,
                title=f"{fn_key}_itr_{itr}",
                yrange=[-0.4, 4],
                loc="upper left",
                normalize=True,
                delta=0.8,
                annotate_rankings=True,
            )

            for other_s, other_k in zip(other_scores, other_keys):
                file = f"key_{self._rng_name}_cand_itr_{itr}_comare_{fn_key}_vs_{other_k}.png"
                path = os.path.join(ranking_dir, file)
                plot_ranking_corrs([cand_scores, other_s], [fn_key, other_k], path=path)
