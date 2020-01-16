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
    * see (rdb.exps.acquire.py) for acquisition functions

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


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        acquire_fns (dict): map name -> acquire function
        model (object): IRD model
        iteration (int): algorithm iterations
        num_eval_sample (int): evaluating belief samples is costly,
            so subsample belief particles
        num_active_tasks (int): # task candidates for active selection
        num_active_sample (int): running acquisition function on belief
            samples is costly, so subsample belief particles

    """

    def __init__(
        self,
        model,
        acquire_fns,
        eval_server,
        iterations=10,
        num_eval_tasks=4,
        num_eval_sample=5,
        num_active_tasks=4,
        num_active_sample=-1,
        num_map_estimate=4,
        fixed_candidates=None,
        fixed_belief_tasks=None,
        save_dir="data/active_ird_exp1",
        exp_name="active_ird_exp1",
    ):
        # Inverse Reward Design Model
        self._model = model
        self._acquire_fns = acquire_fns
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        # self._random_choice = None
        self._random_choice = random_choice
        self._iterations = iterations
        # Evaluation
        self._num_eval_tasks = num_eval_tasks
        self._num_eval_sample = num_eval_sample
        self._num_map_estimate = num_map_estimate
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._fixed_candidates = fixed_candidates
        self._fixed_belief_tasks = fixed_belief_tasks
        # Save path
        self._save_dir = save_dir
        self._exp_name = exp_name
        self._last_time = time()

    def _build_cache(self):
        """ Build cache data """
        self._acquire_eval_hist = {}
        self._all_tasks, self._all_task_names = {}, {}
        self._all_obs, self._all_beliefs = {}, {}
        self._all_candidates = []
        self._eval_tasks = []
        self._curr_belief = {}
        for key in self._acquire_fns.keys():
            self._acquire_eval_hist[key] = []

    def _read_cache(self):
        pass

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        # self._random_choice = seed(random_choice, rng_key)
        self._random_choice = seed(self._random_choice, rng_key)
        for fn in self._acquire_fns.values():
            fn.update_key(rng_key)

    def run(self, task):
        """Main function 1: Run experiment from task.

        Args:
            task (obj): initial task; if None, start from random.

        """
        print(f"\n============= Main Experiment ({self._rng_key}) =============")
        """Run main algorithmic loop."""
        self._build_cache()
        eval_tasks = self._random_choice(
            self._model.env.all_tasks, self._num_eval_tasks
        )
        """ First iteration """
        task_name = f"ird_{str(task)}"
        obs = self._model.simulate_designer(task, task_name)
        belief = self._model.sample([task], [task_name], obs=[obs])
        # self._diagnose_belief(belief, eval_tasks, fn_key="all", itr=0)

        """ Build history for each acquisition function"""
        self._eval_tasks = eval_tasks
        for key in self._acquire_fns.keys():
            self._all_obs[key] = [obs]
            self._all_tasks[key] = [task]
            self._all_task_names[key] = [task_name]
            self._all_beliefs[key] = [belief]
            self._curr_belief[key] = belief
            self._evaluate(key, belief, eval_tasks)
            self._save(itr=0)

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
            for key in self._acquire_fns.keys():
                belief = self._curr_belief[key]
                obs = self._all_obs[key][-1]
                task_name = self._all_task_names[key][-1]
                print(f"Method: {key}; Task name {task_name}")

                ## Actively propose & Record next task
                next_task = self._propose_task(candidates, belief, obs, key)
                next_task_name = f"ird_{str(next_task)}"
                self._all_tasks[key].append(next_task)
                self._all_task_names[key].append(next_task_name)

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

    def debug(self, debug_dir):
        """Main function 2: Debug past experiments from pre-saved directory.

        Args:
            debug_dir (str): checkpoint directory.

        """
        print(
            f"\n============= Debug Experiment ({self._rng_key}): {debug_dir} ============="
        )
        """Run main algorithmic loop."""
        self._load_cache(debug_dir)
        """ First iteration """
        self._eval_tasks = self._random_choice(
            self._model.env.all_tasks, self._num_eval_tasks
        )
        for it in range(0, self._iterations + 1):
            """ Run Active IRD on Candidates """
            print(f"\nEvaluating IRD ({self._rng_key}) iteration {it}")
            for key in self._acquire_fns.keys():
                belief = self._all_beliefs[key][it]
                # Evaluate and Save
                self._evaluate(key, belief, eval_tasks, use_map=self._num_map_estimate)
                self._save(it, skip_weights=True)

    def _propose_task(self, candidates, belief, obs, fn_key):
        """Find best next task for this acquire function.

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
        cand_names = [f"acquire_{task}" for task in candidates]
        belief = belief.subsample(self._num_active_sample)
        belief = self._eval_server.compute_tasks(
            belief, candidates, cand_names, verbose=True
        )

        desc = "Evaluaitng candidate tasks"
        for next_task, next_task_name in tqdm(
            zip(candidates, cand_names), total=len(candidates), desc=desc
        ):
            scores.append(
                self._acquire_fns[fn_key](
                    next_task, next_task_name, belief, obs, verbose=False
                )
            )
        scores = np.array(scores)
        next_task = candidates[np.argmax(scores)]
        return next_task

    def _evaluate(self, fn_name, belief, eval_tasks, use_map=-1):
        """Evaluate current sampled belief on eval task.

        Args:
            use_map: use MAP estimate particle, intead of whole population, to estimate.

        Criteria:
            * Relative Reward.
            * Violations.

        """
        # Compute belief features
        eval_names = [f"eval_{task}" for task in eval_tasks]
        if use_map > 0:
            belief = self._model.create_particles(belief.map_estimate(use_map))
        else:
            belief = belief.subsample(self._num_eval_sample)
        target = self._model.create_particles([self._model.designer.true_w])
        belief = self._eval_server.compute_tasks(
            belief, eval_tasks, eval_names, verbose=True
        )
        target = self._eval_server.compute_tasks(
            target, eval_tasks, eval_names, verbose=True
        )

        num_violate = 0.0
        performance = 0.0
        desc = f"Evaluating method {fn_name}"
        for task, task_name in tqdm(
            zip(eval_tasks, eval_names), total=len(eval_tasks), desc=desc
        ):
            vios = belief.get_violations(task, task_name)
            perf = belief.compare_with(task, task_name, target_particles=target)
            performance += perf.mean()
            num_violate += np.array(list(vios.values())).sum(axis=0).mean()

        avg_violate = float(num_violate / len(eval_tasks))
        avg_perform = float(performance / len(eval_tasks))
        print(f">>> Average Violation {avg_violate:.2f}")
        print(f">>> Average Performance diff {avg_perform:.2f}")
        self._acquire_eval_hist[fn_name].append(
            {"violation": avg_violate, "perform": avg_perform}
        )

    def _load_cache(self, save_dir):
        """Load previous checkpoint."""
        # Load eval data
        eval_path = f"{save_dir}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        eval_data = np.load(eval_path, allow_pickle=True)
        ## Empty eval history
        self._all_candidates = eval_data["candidate_tasks"]
        self._eval_tasks = eval_data["eval_tasks"]
        self._all_tasks = eval_data["curr_tasks"].item()
        np_obs = eval_data["curr_obs"].item()
        self._all_obs = {}
        for key in np_obs.keys():
            self._all_obs[key] = [
                self._model.create_particles([ws]) for ws in np_obs[key]
            ]
        # Load beliefs
        self._all_beliefs = {}
        self._acquire_eval_hist = {}
        weight_dir = f"{save_dir}/{self._exp_name}/save"
        for key in self._acquire_fns.keys():
            self._all_beliefs[key] = []
            self._acquire_eval_hist[key] = []
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

    def _save(self, itr, skip_weights=False):
        """Save checkpoint."""
        exp_dir = f"{self._save_dir}/{self._exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        ## Save evaluation history
        np_obs = {}
        for key in self._all_obs.keys():
            np_obs[key] = [ob.weights[0] for ob in self._all_obs[key]]
        data = dict(
            seed=str(self._rng_key),
            curr_obs=np_obs,
            curr_tasks=self._all_tasks,
            eval_tasks=self._eval_tasks,
            eval_hist=self._acquire_eval_hist,
            candidate_tasks=self._all_candidates,
        )
        path = f"{self._save_dir}/{self._exp_name}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        with open(path, "wb+") as f:
            np.savez(f, **data)

        ## Save belief sample information
        fig_dir = f"{self._save_dir}/{self._exp_name}/plots"
        save_dir = f"{self._save_dir}/{self._exp_name}/save"
        os.makedirs(fig_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        for key in self._all_beliefs.keys():
            for itr, belief in enumerate(self._all_beliefs[key]):
                fname = f"weights_seed_{str(self._rng_key)}_method_{key}_itr_{itr:02d}"
                savepath = f"{save_dir}/{fname}.npz"
                figpath = f"{fig_dir}/{fname}.png"
                if itr < len(np_obs[key]):
                    obs_w = np_obs[key][itr]
                else:
                    obs_w = None
                if not skip_weights:
                    belief.save(savepath)
                    belief.visualize(
                        figpath, true_w=self._model.designer.true_w, obs_w=obs_w
                    )
        self._log_time()

    def _log_time(self):
        if self._last_time is not None:
            secs = time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f"Active IRD Iteration Time: {int(h)}h {int(m)}m {s:.3f}s\n")
        self._last_time = time()

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
