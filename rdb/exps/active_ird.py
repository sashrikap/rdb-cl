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

Note:
    * see (rdb.exps.active.py) for acquisition functions

"""

from rdb.infer.utils import random_choice
from rdb.infer.particles import Particles
from rdb.visualize.plot import *
from rdb.exps.utils import *
from numpyro.handlers import seed
from tqdm.auto import tqdm
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
        fixed_candidates=None,
        fixed_belief_tasks=None,
        save_dir="data/active_ird_exp1",
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
        # self._random_choice = random_choice
        self._iterations = iterations
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
        self._save_dir = save_dir
        self._exp_name = exp_name
        self._last_time = time()

    def _build_cache(self):
        """ Build cache data """
        self._active_eval_hist = {}
        self._all_tasks, self._all_task_names = {}, {}
        self._all_obs, self._all_beliefs = {}, {}
        self._all_candidates = []
        self._all_cand_scores = {}
        self._eval_tasks = []
        self._curr_belief = {}
        for key in self._active_fns.keys():
            self._active_eval_hist[key] = []

    def _read_cache(self):
        pass

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        self._random_choice = seed(random_choice, rng_key)
        for fn in self._active_fns.values():
            fn.update_key(rng_key)

    def run(self):
        """Main function 1: Run experiment from task.

        Args:
            task (obj): initial task; if None, start from random.

        """
        print(
            f"\n============= Main Experiment ({self._rng_key}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        """Run main algorithmic loop."""
        self._build_cache()
        eval_tasks = self._random_choice(
            self._model.env.all_tasks, self._num_eval_tasks
        )
        if self._num_eval_tasks > 0:
            self._eval_tasks = eval_tasks
        else:
            self._eval_tasks = []  # do not save when eval onall tasks

        """ First iteration """
        task = self._random_choice(self._model.env.all_tasks, 1)[0]
        task_name = f"ird_{str(task)}"
        obs = self._model.simulate_designer(task, task_name)
        belief = self._model.sample([task], [task_name], obs=[obs])

        """ Build history for each acquisition function"""
        for key in self._active_fns.keys():
            self._all_obs[key] = [obs]
            self._all_tasks[key] = [task]
            self._all_task_names[key] = [task_name]
            self._all_beliefs[key] = [belief]
            self._all_cand_scores[key] = []
            self._curr_belief[key] = belief
            self._evaluate(key, belief, eval_tasks)
            self._save(itr=0)
            self._log_time("Itr 0 Eval & Save")

        for it in range(1, self._iterations + 1):
            """ Candidates for next tasks """
            if self._fixed_candidates is not None:
                candidates = self._fixed_candidates
            else:
                candidates = self._random_choice(
                    self._model.env.all_tasks, self._num_active_tasks
                )
            self._all_candidates.append(candidates)

            """ Run Active IRD on Candidates """
            print(f"\nActive IRD ({self._rng_key}) iteration {it}")
            for key in self._active_fns.keys():
                belief = self._curr_belief[key]
                obs = self._all_obs[key][-1]
                task_name = self._all_task_names[key][-1]
                print(f"Method: {key}; Task name {task_name}")

                ## Actively propose & Record next task
                next_task, scores = self._propose_task(candidates, belief, obs, key)
                next_task_name = f"ird_{str(next_task)}"
                self._all_tasks[key].append(next_task)
                self._all_task_names[key].append(next_task_name)
                self._all_cand_scores[key].append(scores)
                self._log_time(f"Itr {it} {key} Propose")

                ## Main IRD Sampling
                next_obs = self._model.simulate_designer(next_task, next_task_name)
                next_belief = self._model.sample(
                    self._all_tasks[key],
                    self._all_task_names[key],
                    obs=self._all_obs[key],
                )
                self._all_obs[key].append(next_obs)
                self._all_beliefs[key].append(next_belief)
                self._curr_belief[key] = next_belief
                # Evaluate and Save
                self._evaluate(key, next_belief, eval_tasks)
                self._save(it)
                self._log_time(f"Itr {it} {key} Eval & Save")

    def _propose_task(self, candidates, belief, obs, fn_key):
        """Find best next task for this active function.

        Args:
            candidates (list): potential next tasks
            obs (Particles.weights[1]): current observed weight
            fn_key (str): acquisition function key

        Note:
            * Require `tasks = env.sample_task()`
            * Use small task space to avoid being too slow.

        """
        scores = []
        # Compute belief features
        cand_names = [f"active_{task}" for task in candidates]
        belief = belief.subsample(self._num_active_sample)
        if fn_key != "random":
            # Random does not need pre-computation
            self._eval_server.compute_tasks(
                belief, candidates, cand_names, verbose=True
            )
            self._eval_server.compute_tasks(obs, candidates, cand_names, verbose=True)

        desc = "Evaluaitng candidate tasks"
        for next_task, next_task_name in tqdm(
            zip(candidates, cand_names), total=len(candidates), desc=desc
        ):
            scores.append(
                self._active_fns[fn_key](
                    next_task, next_task_name, belief, obs, verbose=False
                )
            )
        scores = np.array(scores)
        next_task = candidates[np.argmax(scores)]
        return next_task, scores

    def _evaluate(self, fn_name, belief, eval_tasks):
        """Evaluate current sampled belief on eval task.

        Note:
            self._num_eval_map: use MAP estimate particle, intead of whole population, to estimate.

        Criteria:
            * Relative Reward.
            * Violations.

        """
        if self._model.interactive_mode:
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
            print(f">>> Average Violation diff {avg_violate:.2f}")
            print(f">>> Average Performance diff {avg_perform:.2f}")
        self._active_eval_hist[fn_name].append(
            {"violation": avg_violate, "perform": avg_perform}
        )

    def _load_cache(self, load_dir, load_eval=True):
        """Load previous checkpoint."""
        # Load eval data
        eval_path = f"{load_dir}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        eval_data = np.load(eval_path, allow_pickle=True)
        if load_eval:
            self._active_eval_hist = eval_data["eval_hist"].item()
        else:
            self._active_eval_hist = {}
        self._all_candidates = eval_data["candidate_tasks"]
        self._all_cand_scores = eval_data["candidate_scores"].item()
        self._eval_tasks = eval_data["eval_tasks"]
        if len(self._eval_tasks) == 0:
            self._eval_tasks = self._model.env.all_tasks
        self._all_tasks = eval_data["curr_tasks"].item()

        # Load observations
        np_obs = eval_data["curr_obs"].item()
        self._all_obs = {}
        for key in np_obs.keys():
            self._all_obs[key] = [
                self._model.create_particles([ws]) for ws in np_obs[key]
            ]
        if "env_id" in eval_data:
            assert self._model.env_id == eval_data["env_id"].item()
        # Load truth
        if "truth" in eval_data:
            true_ws = [eval_data["true_w"].item()]
            self._model.designer.truth.weights = true_ws
        # Load parameters and check
        # if "exp_params" in eval_data:
        #     exp_params = eval_data["exp_params"].item()
        #     for key, val in exp_params.items():
        #         assert key in self._exp_params and self._exp_params == val

        # Load beliefs
        self._all_beliefs = {}
        weight_dir = f"{load_dir}/{self._exp_name}/save"
        for key in self._active_fns.keys():
            self._all_beliefs[key] = []
            self._active_eval_hist[key] = []
            weight_files = sorted(
                [
                    f
                    for f in os.listdir(weight_dir)
                    if key in f and str(self._rng_key) in f
                ]
            )
            for file in weight_files:
                weight_filepath = os.path.join(weight_dir, file)
                belief = self._model.create_particles([])
                belief.load(weight_filepath)
                self._all_beliefs[key].append(belief)

        # Load random seed
        rng_key = str_to_key(eval_data["seed"].item())
        self.update_key(rng_key)

    def _save(self, itr, skip_weights=False):
        """Save checkpoint.

        Format:
            * data/save_dir/exp_name/{exp_name}_seed.npz
              - seed (str): rng_key
              - curr_obs (dict): {method: [obs_w] * num_itr}
              - curr_tasks (dict): {method: [task] * num_eval}
              - eval_tasks (dict): {method: [task] * num_eval}
              - eval_hist (dict): {method: [
                    {"violation": ..., "perform": ...}
                )] * num_itr}
            * data/save_dir/exp_name/save/weights_seed_method_itr.npz
              - see `rdb.infer.particles.save()`

        """
        exp_dir = f"{self._save_dir}/{self._exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
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
            eval_tasks=self._eval_tasks,
            eval_hist=self._active_eval_hist,
            candidate_tasks=self._all_candidates,
            candidate_scores=self._all_cand_scores,
        )
        path = f"{self._save_dir}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        with open(path, "wb+") as f:
            np.savez(f, **data)
        print("save", exp_dir)

        ## Save belief sample information
        fig_dir = f"{self._save_dir}/{self._exp_name}/plots"
        save_dir = f"{self._save_dir}/{self._exp_name}/save"
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        for key in self._all_beliefs.keys():
            for itr, belief in enumerate(self._all_beliefs[key]):
                fname = f"weights_seed_{str(self._rng_key)}_method_{key}_itr_{itr:02d}"
                savepath = f"{save_dir}/{fname}.npz"
                figpath = f"{fig_dir}/{fname}"
                if itr < len(np_obs[key]):
                    obs_w = np_obs[key][itr]
                else:
                    obs_w = None
                if not skip_weights:
                    belief.save(savepath)
                    belief.visualize(
                        figpath, true_w=self._model.designer.true_w, obs_w=obs_w
                    )

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f"Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time()

    def debug_candidate(self, debug_dir):
        """Debug past experiments' candidates from pre-saved directory.

        Steps:
            * Plots performance curve
            * For each iteration, for each method: (1) visualize proposal scores on
            candidate tasks vs other methods (2) show mp4 for top 5/worst 5/map 5 belief
            points on candidate tasks

        """
        print(
            f"\n============= Debug Candidates ({self._rng_key}): {debug_dir}{self._exp_name} ============="
        )
        self._load_cache(debug_dir)
        all_cands = self._all_candidates
        all_scores = self._all_cand_scores
        all_beliefs = self._all_beliefs

        num_iter = all_cands.shape[0]
        ranking_dir = os.path.join(debug_dir, self._exp_name, "candidates")
        video_dir = os.path.join(debug_dir, self._exp_name, "cand_video")
        os.makedirs(ranking_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        truth = self._model.designer.truth
        for itr in range(num_iter):
            candidates = all_cands[itr]
            for fn_key in all_scores.keys():
                # Check current active function
                if len(all_scores[fn_key]) <= itr:
                    continue
                cand_scores = onp.array(all_scores[fn_key][itr])

                # Check other active function
                other_scores_all = copy.deepcopy(all_scores)
                del other_scores_all[fn_key]
                other_scores = []
                other_keys = []
                for key in other_scores_all.keys():
                    if len(other_scores_all[key]) > itr:
                        other_scores.append(other_scores_all[key][itr])
                        other_keys.append(key)

                # Ranking plot
                file = f"key_{self._rng_key}_cand_itr_{itr}_fn_{fn_key}_ranking.png"
                path = os.path.join(ranking_dir, file)
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
                        video=True,
                    )

    # def _diagnose_belief(self, belief, tasks, fn_key, itr):
    #     """Diagnose proxy reward over evaluation tasks.

    #     Diagnose principle:
    #         * Look at 5 top, 5 worst and 5 medium belief particles.

    #     """
    #     diagnose_N = 5
    #     debug_dir = (
    #         f"{self._save_dir}/{self._exp_name}/diagnose_seed_{str(self._rng_key)}"
    #     )
    #     os.makedirs(debug_dir, exist_ok=True)
    #     for t_i, task in tqdm(enumerate(tasks), total=len(tasks), desc="Diagnosing"):
    #         task_name = f"debug_{str(task)}"
    #         file_prefix = f"{debug_dir}/fn_{fn_key}_itr_{itr}_"
    #         belief.diagnose(
    #             task, task_name, self._model.designer.true_w, diagnose_N, file_prefix
    #         )
