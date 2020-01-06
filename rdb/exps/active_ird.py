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
from numpyro.handlers import seed
from tqdm.auto import tqdm
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
        fixed_candidates=None,
        debug_belief_task=None,
        save_dir="data/active_ird_exp1",
        exp_name="active_ird_exp1",
    ):
        # Inverse Reward Design Model
        self._model = model
        self._acquire_fns = acquire_fns
        self._eval_server = eval_server
        # Random key & function
        self._rng_key = None
        self._random_choice = random_choice
        self._iterations = iterations
        # Evaluation
        self._num_eval_tasks = num_eval_tasks
        self._num_eval_sample = num_eval_sample
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._fixed_candidates = fixed_candidates
        self._debug_belief_task = debug_belief_task
        # Save path
        self._save_dir = save_dir
        self._exp_name = exp_name

    def _build_cache(self):
        """ Build cache data """
        self._acquire_eval_hist = {}
        self._all_tasks, self._all_task_names = {}, {}
        self._all_obs, self._all_beliefs = {}, {}
        self._curr_belief = {}
        for key in self._acquire_fns.keys():
            self._acquire_eval_hist[key] = []

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        self._random_choice = seed(self._random_choice, rng_key)
        for fn in self._acquire_fns.values():
            fn.update_key(rng_key)

    def run(self, task):
        """Run experiment from task.

        Args:
            task (obj): initial task; if None, start from random.

        """
        print(f"\n============= Main Experiment ({self._rng_key}) =============")
        """Run main algorithmic loop."""
        self._build_cache()
        eval_tasks = self._random_choice(
            self._model.env.all_tasks, self._num_eval_tasks, replacement=True
        )
        """ First iteration """
        task_name = f"ird_{str(task)}"
        obs = self._model.simulate_designer(task, task_name)
        belief = self._model.sample([task], [task_name], obs=[obs], visualize=True)

        """ Build history for each acquisition function"""
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
            """ Run Active IRD on Candidates """
            print(f"\nActive IRD iteration {it}")
            for key in self._acquire_fns.keys():
                ## Evaluate
                # if self._debug_belief_task is not None:
                #    self._debug_belief(belief, self._debug_belief_task, obs)
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
                    visualize=True,
                )
                self._all_obs[key].append(next_obs)
                self._all_beliefs[key].append(next_belief)
                self._curr_belief[key] = next_belief
                # Evaluate and Save
                self._evaluate(key, belief, eval_tasks)
                self._save(it)

        # Evaluate and save last round
        for key in self._acquire_fns.keys():
            belief = self._curr_belief[key]
            self._evaluate(key, belief, eval_tasks)
        self._save(self._iterations)

    def _debug_belief(self, belief, task, obs):
        print(f"Observed weights")
        for key, val in obs.weights[0].items():
            print(f"-> {key}: {val:.3f}")
        task_name = f"debug_{str(task)}"
        feats = belief.get_features(task, task_name)
        violations = belief.get_violations(task, task_name)
        # if np.sum(info["metadata"]["overtake1"]) > 0:
        #    import pdb; pdb.set_trace()

    def _propose_task(self, candidates, belief, obs, fn_key):
        """Find best next task for this acquire function.

        Args:
            candidates (list): potential next tasks
            obs (Particle[1]): current observed weight
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
            # print(f"Task candidate {ni+1}/{len(candidates)}")
            scores.append(
                self._acquire_fns[fn_key](
                    next_task, next_task_name, belief, obs, verbose=False
                )
            )
        scores = np.array(scores)
        next_task = candidates[np.argmax(scores)]
        return next_task

    def _evaluate(self, fn_name, belief, eval_tasks):
        """Evaluate current sampled belief on eval task.

        Criteria:
            * Relative Reward.
            * Violations.

        """
        # COmpute belief features
        eval_names = [f"eval_{task}" for task in eval_tasks]
        belief = belief.subsample(self._num_eval_sample)
        belief = self._eval_server.compute_tasks(
            belief, eval_tasks, eval_names, verbose=True
        )

        num_violate = 0.0
        performance = 0.0
        desc = f"Evaluating method {fn_name}"
        for task, task_name in tqdm(
            zip(eval_tasks, eval_names), total=len(eval_tasks), desc=desc
        ):
            vios = belief.get_violations(task, task_name)
            perf = belief.compare_with(task, task_name, self._model.designer.true_w)

            performance += perf
            num_violate += np.array(list(vios.values())).mean()

        avg_violate = float(num_violate / len(eval_tasks))
        avg_perform = float(performance / len(eval_tasks))
        print(f">>> Average Violation {avg_violate:.2f}")
        print(f">>> Average Performance diff {avg_perform:.2f}")
        self._acquire_eval_hist[fn_name].append(
            {"violation": avg_violate, "perform": avg_perform}
        )

    def _save(self, itr):
        expdir = f"{self._save_dir}/{self._exp_name}"
        os.makedirs(expdir, exist_ok=True)
        ## Save evaluation history
        np_obs = {}
        for key in self._all_obs.keys():
            np_obs[key] = [ob.weights[0] for ob in self._all_obs[key]]
        data = dict(
            curr_obs=np_obs,
            curr_tasks=self._all_tasks,
            eval_hist=self._acquire_eval_hist,
        )
        path = f"{self._save_dir}/{self._exp_name}_seed_{str(self._rng_key)}.npz"
        with open(path, "wb+") as f:
            np.savez(f, **data)

        ## Save belief sample information
        figdir = f"{self._save_dir}/{self._exp_name}/plots"
        savedir = f"{self._save_dir}/{self._exp_name}/save"
        os.makedirs(figdir, exist_ok=True)
        os.makedirs(savedir, exist_ok=True)
        for key in self._all_beliefs.keys():
            for itr, belief in enumerate(self._all_beliefs[key]):
                fname = f"weights_seed_{str(self._rng_key)}_method_{key}_itr_{itr:02d}"
                savepath = f"{savedir}/{fname}.npz"
                figpath = f"{figdir}/{fname}.png"
                belief.save(savepath)
                belief.visualize(figpath, self._model.designer.true_w)
