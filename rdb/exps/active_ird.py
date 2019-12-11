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
import jax.numpy as np
import copy


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        acquire_fns (dict): map name -> acquire function
        model (object): IRD model

    """

    def __init__(
        self,
        env,
        true_w,
        model,
        acquire_fns,
        eval_tasks,
        iterations=10,
        num_eval=20,
        num_task_sample=4,
    ):
        self._env = env
        self._true_w = true_w
        self._model = model
        self._acquire_fns = acquire_fns
        self._eval_tasks = eval_tasks
        # Random key & function
        self._rng_key = None
        self._random_choice = random_choice
        # Numerics
        self._iterations = iterations
        self._num_eval = num_eval
        self._num_task_sample = num_task_sample
        # Cache data
        self._acquire_samples = {}
        for key in acquire_fns.keys():
            self._acquire_samples[key] = None

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        self._random_choice = seed(self._random_choice, rng_key)

    def run(self, task):
        """Run main evaluation loop."""
        curr_tasks = {}
        for fn_key in self._acquire_fns.keys():
            # Build history for each acquisition function
            curr_tasks[fn_key] = [task]

        for it in range(self._iterations):
            print(f"\nActive IRD iteration {it}")
            for fn_key, fn_tasks in curr_tasks.items():
                """ Collect observation """
                task = fn_tasks[-1]
                obs_w = self._simulate_designer(task)
                """ Collect features for task and cache """
                task_name = f"ird_{str(task)}"
                print(f"Fn name: {fn_key}; Task name {task_name}")
                if it > 0:
                    self._model.sample_features(task, task_name)
                """ Assign task """
                belief_ws = self._model.sample(task, task_name, obs_w=obs_w)
                _, belief_feats = self._model.get_samples(task, task_name)
                """ Evaluate """
                self._evaluate(belief_w)
                """ Actively propose next task """
                fn_next_task = self._propose_task()
                curr_tasks[fn_key] = fn_next_task

    def _simulate_designer(self, task):
        """Sample one w from b(w) on task"""
        task_name = f"designer_{str(task)}"
        designer_ws = self._model.sample_designer(task, task_name, self._true_w)
        designer_w = self._random_choice(designer_ws, 1)
        return designer_w[0]

    def _propose_task(self):
        """Propose next task to designer.

        Note:
            * Require `tasks = env.sample_task()`
            * Use small task space to avoid being too slow.

        TODO:
            * `env.sample_task()`: small number of samples
        """
        proposed_tasks = self._env.sample_task(self._num_task_sample)
        acquire_scores = {}
        acquire_tasks = {}
        for task in proposed_tasks:
            for key, ac_fn in self._acquire_fns.items():
                acquire_scores[key] = ac_fn(task)

        """ Find best task for each acquire function """
        for key, scores in acquire_scores.items():
            acquire_tasks[key] = proposed_tasks[np.argsort(scores)[0]]

    def _evaluate(self, belief_w):
        """Evaluate current sampled belief.

        Note:
            * Require `runner` to collect constraint violations.
            * The fewer violations, the better.

        """
        # violations =
        # Find the top M belief samples and average violations
        num_violate = 0.0
        for task in self._eval_tasks:
            self._env.set_task(task)
            self._env.reset()
            state = self._env.state
            actions = self._controller(state, weights=weights)
            traj, cost, info = self._runner(state, actions, weights=weights)
            violations = info["violations"]
            num_violate += sum([sum(v) for v in violations.values()])
        print(f"Average Violation {num_violate / len(self._eval_tasks):.2f}")
