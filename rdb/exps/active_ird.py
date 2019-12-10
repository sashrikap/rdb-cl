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
        # Numerics
        self._iterations = iterations
        self._num_eval = num_eval
        self._num_task_sample = num_task_sample
        # Cache data
        self._acquire_samples = {}
        for key in acquire_fns.keys():
            self._acquire_samples[key] = None

    def run(self, task):
        """Run main evaluation loop."""
        curr_tasks = {}
        curr_models = {}
        for fn_key in self._acquire_fns.keys():
            # Build history for each acquisition function
            curr_tasks[fn_key] = task
            curr_models[fn_key] = copy.deepcopy(self._model)

        for it in range(self._iterations):
            print(f"Active IRD iteration {it}")
            for fn_key, fn_task in curr_tasks.items():
                """ Collect features for task and cache """
                model = curr_models[fn_key]
                task_name = str(fn_task)
                if it > 0:
                    model.sample_features(fn_task, task_name)
                """ Assign task """
                self._env.set_task(fn_task)
                model.initialize(fn_task, task_name)
                """ Current belief """
                b_w = model.sample()
                """ Evaluate """
                self._evaluate(b_w)
                """ Actively propose next task """
                fn_next_task = self._propose_task()
                curr_tasks[fn_key] = fn_next_task

    def _simulate_designer(self, sampler, itr):
        """Sample 1 weight from belief samples.

        Args:
            sampler (object): probabilistic sampler for current task

        Note:
            * Practically only need to run batch sampling once
            * Cache batch samples

        TODO:
            * random choice from `sampler.sample()`
        """
        pass

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
