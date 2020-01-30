"""Active-IRD Experiment.

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
import os


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        active_fns (dict): map name -> active function
        model (object): IRD model
        iteration (int): algorithm iterations
        num_eval_map (int): if > 0, use MAP estimate
        num_eval_sample (int): if num_eval_map=-1, uniformly subsample current belief
        num_active_tasks (int): # task candidates for active selection
        num_active_sample (int): running acquisition function on belief
            samples is costly, so subsample belief particles

    """

    def __init__(
        self,
        model,
        active_fns,
        eval_server,
        iterations=10,
        num_eval_tasks=4,
        num_eval_map=-1,
        num_eval_sample=5,
        num_active_tasks=4,
        num_active_sample=-1,
        fixed_task_seed=None,
        fixed_candidates=None,
        fixed_belief_tasks=None,
        normalized_key=None,
        design_data={},
        num_load_design=-1,
        save_root="data/active_ird_exp1",
        exp_name="active_ird_exp1",
        exp_params={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._active_fns = active_fns
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        self._random_choice = None
        self._random_task_choice = None
        self._fixed_task_seed = fixed_task_seed
        self._iterations = iterations
        self._normalized_key = normalized_key
        # Evaluation
        self._num_eval_map = num_eval_map
        self._num_eval_tasks = num_eval_tasks
        self._num_eval_sample = num_eval_sample
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._fixed_candidates = fixed_candidates
        self._num_active_sample = num_active_sample
        self._fixed_belief_tasks = fixed_belief_tasks
        # Save path
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._last_time = time()
        # Load design and cache
        self._num_load_design = num_load_design
        self._design_data = design_data

    def _build_cache(self):
        """ Build cache data """
        self._initial_task = None
        self._active_eval_hist = {}
        self._all_tasks, self._all_task_names = {}, {}
        self._all_obs, self._all_beliefs = {}, {}
        self._all_candidates = []
        self._all_cand_scores = {}
        self._eval_tasks = []
        self._curr_belief = {}
        for key in self._active_fns.keys():
            self._active_eval_hist[key] = []
            self._all_cand_scores[key] = []
            self._all_tasks[key], self._all_task_names[key] = [], []
            self._all_obs[key], self._all_beliefs[key] = [], []

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        if self._fixed_task_seed is not None:
            self._random_task_choice = seed(random_choice, self._fixed_task_seed)
        else:
            self._random_task_choice = self._random_choice
        for fn in self._active_fns.values():
            fn.update_key(rng_key)

    def run(self, plot_candidates=False):
        """Main function 1: Run experiment from task.

        Args:
            # task (obj): initial task; if None, start from random.
            plot_candidates (bool): plot rankings of proposed candidates

        """
        print(
            f"\n============= Main Experiment ({self._rng_key}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        self._build_cache()
        self._load_design()
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(self._model.env.all_tasks):
            num_eval = -1
        self._eval_tasks = self._random_task_choice(self._model.env.all_tasks, num_eval)

        ### Main Experiment Loop ###
        for itr in range(0, self._iterations):
            ## Candidates for next tasks
            if self._fixed_candidates is not None:
                candidates = self._fixed_candidates
            else:
                candidates = self._random_task_choice(
                    self._model.env.all_tasks, self._num_active_tasks
                )
            self._all_candidates.append(candidates)

            ### Run Active IRD on Candidates ###
            print(f"\nActive IRD ({self._rng_key}) iteration {itr}")
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

                task_name = f"ird_{str(task)}"
                self._all_tasks[key].append(task)
                self._all_task_names[key].append(task_name)
                self._all_cand_scores[key].append(scores)
                self._log_time(f"Itr {itr} {key} Propose")

                ## Simulate Designer
                obs = self._model.simulate_designer(
                    task=task,
                    task_name=task_name,
                    save_name=f"designer_seed_{str(self._rng_key)}_method_{key}_itr_{itr:02d}",
                )
                self._all_obs[key].append(obs)
                self._log_time(f"Itr {itr} {key} Designer")

                ## IRD Sampling w/ Divide And Conquer
                belief = self._model.sample(
                    tasks=self._all_tasks[key],
                    task_names=self._all_task_names[key],
                    obs=self._all_obs[key],
                    save_name=f"weights_seed_{str(self._rng_key)}_method_{key}_itr_{itr:02d}",
                )
                self._all_beliefs[key].append(belief)
                self._curr_belief[key] = belief
                self._save(itr=itr)
                self._log_time(f"Itr {itr} {key} IRD Divide & Conquer")

                ## Evaluate, plot and Save
                self._plot_candidate_scores(itr)
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
            task = self._random_task_choice(self._model.env.all_tasks, 1)[0]
            self._initial_task = task
        else:
            task = self._initial_task
        scores = onp.zeros(self._num_active_tasks)
        return task, scores

    def _propose_task(self, candidates, all_beliefs, all_obs, all_tasks, fn_key):
        """Find best next task for this active function.

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
        cand_names = [f"active_{task}" for task in candidates]
        belief = all_beliefs[-1]
        belief = belief.subsample(self._num_active_sample)
        if fn_key != "random":
            # Random does not need pre-computation
            self._eval_server.compute_tasks(
                belief, candidates, cand_names, verbose=True
            )
            self._eval_server.compute_tasks(
                all_obs[-1], candidates, cand_names, verbose=True
            )

        scores = []
        desc = "Evaluaitng candidate tasks"
        for next_task, next_task_name in tqdm(
            zip(candidates, cand_names), total=len(candidates), desc=desc
        ):
            scores.append(
                self._active_fns[fn_key](
                    next_task, next_task_name, belief, all_obs, verbose=False
                )
            )
        scores = onp.array(scores)
        print(f"Function {fn_key} chose task {onp.argmax(scores)} among {len(scores)}")
        next_task = candidates[onp.argmax(scores)]
        return next_task, scores

    def _evaluate(self, fn_name, belief, eval_tasks):
        """Evaluate current sampled belief on eval task.

        Note:
            self._num_eval_map: use MAP estimate particle, intead of whole population, to estimate.

        Criteria:
            * Relative Reward.
            * Violations.

        """
        if self._model.interactive_mode and self._model.designer.run_from_ipython():
            # Interactive mode, skip evaluation to speed up
            avg_violate = 0.0
            avg_perform = 0.0
        else:
            # Compute belief features
            eval_names = [f"eval_{task}" for task in eval_tasks]
            if self._num_eval_map > 0:
                belief = belief.map_estimate(self._num_eval_map)
            else:
                belief = belief.subsample(self._num_eval_sample)
            target = self._model.designer.truth
            self._eval_server.compute_tasks(
                belief, eval_tasks, eval_names, verbose=True
            )
            self._eval_server.compute_tasks(
                target, eval_tasks, eval_names, verbose=True
            )

            num_violate = 0.0
            performance = 0.0
            desc = f"Evaluating method {fn_name}"
            for task, task_name in tqdm(
                zip(eval_tasks, eval_names), total=len(eval_tasks), desc=desc
            ):
                diff_perf, diff_vios = belief.compare_with(
                    task, task_name, target=target
                )
                performance += diff_perf.mean()
                num_violate += diff_vios.mean()

            avg_violate = float(num_violate / len(eval_tasks))
            avg_perform = float(performance / len(eval_tasks))
            print(f"    Average Violation diff {avg_violate:.2f}")
            print(f"    Average Performance diff {avg_perform:.2f}")
        self._active_eval_hist[fn_name].append(
            {"violation": avg_violate, "perform": avg_perform}
        )

    def _load_design(self):
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
        num_load, design_data = self._num_load_design, self._design_data
        if num_load < 0:
            return
        assert (
            design_data["ENV_NAME"] == self._exp_params["ENV_NAME"]
        ), "Environment name mismatch"
        assert len(design_data["DESIGNS"]) >= num_load, "Not enough designs to load"

        load_designs = design_data["DESIGNS"][:num_load]
        for i, design_i in enumerate(load_designs):
            # Load previous design
            task = design_i["TASK"]
            task_name = f"design_{task}"
            weights = normalize_weights(design_i["WEIGHTS"], self._normalized_key)
            obs = self._model.create_particles(
                [weights],
                save_name=f"weights_seed_{str(self._rng_key)}_prior_design_{i:02d}",
            )

            # Cache previous designs for active functions
            for key in self._active_fns.keys():
                # Make sure nothing cached before design data
                assert len(self._all_obs[key]) == i
                assert len(self._all_tasks[key]) == i
                assert len(self._all_task_names[key]) == i
                self._all_obs[key].append(obs)
                self._all_tasks[key].append(task)
                self._all_task_names[key].append(task_name)

        print(f"Loaded {len(load_designs)} prior designs.")
        ## Compute IRD Belief based on loaded data
        belief = self._model.sample(
            self._all_tasks[key],
            self._all_task_names[key],
            obs=self._all_obs[key],
            save_name=f"weights_seed_{str(self._rng_key)}_prior_belief",
        )
        for i in range(num_load):
            # Pack the same beliefs into belief history
            assert len(self._all_beliefs[key]) == i
            self._all_beliefs[key].append(belief)

    def _load_cache(self, load_dir, load_eval=True):
        """Load previous experiment checkpoint.

        The opposite ove self._save().

        """
        # Load eval data
        self._build_cache()
        eval_path = f"{load_dir}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        if not os.path.isfile(eval_path):
            print(f"Failed to load {eval_path}")
            return False

        eval_data = np.load(eval_path, allow_pickle=True)
        if load_eval:
            self._active_eval_hist = eval_data["eval_hist"].item()
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
                self._model.create_particles([ws], save_name="observation")
                for ws in np_obs[key]
            ]
        if "env_id" in eval_data:
            assert self._model.env_id == eval_data["env_id"].item()
        # Load truth
        if "truth" in eval_data:
            true_ws = [eval_data["true_w"].item()]
            self._model.designer.truth.weights = true_ws
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
                [
                    f
                    for f in os.listdir(weight_dir)
                    if key in f and str(self._rng_key) in f
                ]
            )
            for file in weight_files:
                weight_filepath = os.path.join(weight_dir, file)
                belief = self._model.create_particles([], save_name="belief")
                belief.load(weight_filepath)
                self._all_beliefs[key].append(belief)
        # Load random seed
        rng_key = str_to_key(eval_data["seed"].item())
        self.update_key(rng_key)
        return True

    def _save(self, itr, skip_weights=False):
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
        exp_dir = f"{self._save_root}/{self._exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        ## Save experiment parameters
        save_params(f"{exp_dir}/params_{str(self._rng_key)}.yaml", self._exp_params)
        ## Save evaluation history
        np_obs = {}
        for key in self._all_obs.keys():
            np_obs[key] = [ob.weights[0] for ob in self._all_obs[key]]
        data = dict(
            seed=str(self._rng_key),
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            true_w=self._model.designer.true_w,
            curr_obs=np_obs,
            curr_tasks=self._all_tasks,
            eval_hist=self._active_eval_hist,
            candidate_tasks=self._all_candidates,
            candidate_scores=self._all_cand_scores,
            eval_tasks=self._eval_tasks
            if self._num_eval_tasks > 0
            else [],  # do not save when eval onall tasks (too large)
        )
        path = f"{self._save_root}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        with open(path, "wb+") as f:
            np.savez(f, **data)
        print("save", exp_dir)

        ## Save belief sample information
        true_w = self._model.designer.true_w
        for key in self._all_beliefs.keys():
            for itr, belief in enumerate(self._all_beliefs[key]):
                if itr < len(np_obs[key]):
                    obs_w = np_obs[key][itr]
                else:
                    obs_w = None
                if not skip_weights:
                    belief.save()
                    belief.visualize(true_w=true_w, obs_w=obs_w)

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time()

    def run_evaluation(self, override=False):
        """For interactive mode. Since it's costly to run evaluation during interactive
        mode, we save the beliefs and evaluate post-hoc.

        Args:
            override (bool): if false, do not re-calculate evaluations.

        """
        print(
            f"\n============= Evaluate Candidates ({self._rng_key}): {self._save_root} {self._exp_name} ============="
        )
        if not self._load_cache(self._save_root):
            return
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
                if override or len(self._active_eval_hist[key]) <= itr:
                    # self._evaluate(key, belief, eval_tasks)
                    self._save(itr=itr)
                    self._log_time(f"Itr {itr} Method {key} Eval & Save")

    def debug_candidate(self, debug_dir):
        """Debug past experiments' candidates from pre-saved directory.

        Steps:
            * Plots performance curve
            * For each iteration, for each method: (1) visualize proposal scores on
            candidate tasks vs other methods (2) show mp4 for top 5/worst 5/map 5 belief
            points on candidate tasks

        """
        print(
            f"\n============= Debug Candidates ({self._rng_key}): {debug_dir} {self._exp_name} ============="
        )
        if not self._load_cache(debug_dir):
            print(f"Failed to load from {debug_dir}")
            return
        all_cands = self._all_candidates
        all_scores = self._all_cand_scores
        all_beliefs = self._all_beliefs
        num_iter = all_cands.shape[0]

        video_dir = os.path.join(debug_dir, self._exp_name, "cand_video")
        os.makedirs(video_dir, exist_ok=True)
        truth = self._model.designer.truth
        for itr in range(0, num_iter):
            self._plot_candidate_scores(itr, debug_dir)
            candidates = all_cands[itr]
            for fn_key in all_scores.keys():
                # Save thumbnail & videos
                top_N = 3
                top_tasks = onp.argsort(-1 * cand_scores)[:top_N]
                belief = all_beliefs[fn_key][itr]
                for ranki in range(top_N):
                    task = candidates[top_tasks[ranki]]
                    task_name = f"cand_{task}"
                    prefix = f"{video_dir}/key_{self._rng_key}_cand_itr_{itr}_fn_{fn_key}_rank_{ranki}_task_{top_tasks[ranki]}_"
                    belief.diagnose(
                        task,
                        task_name,
                        truth,
                        diagnose_N=3,
                        prefix=prefix,
                        thumbnail=True,
                        video=False,
                    )

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
            file = f"key_{self._rng_key}_cand_itr_{itr}_fn_{fn_key}_ranking.png"
            path = os.path.join(ranking_dir, file)
            print(f"Candidate plot saved to {path}")
            plot_rankings(
                cand_scores,
                fn_key,
                other_scores,
                other_keys,
                path=path,
                title=f"{fn_key}_itr_{itr}",
                yrange=[-1.2, 1.6],
                loc="upper left",
                normalize=True,
                delta=0.2,
                annotate_rankings=True,
            )
