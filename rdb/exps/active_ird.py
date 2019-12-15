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
import copy


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        acquire_fns (dict): map name -> acquire function
        model (object): IRD model
        iteration (int): algorithm iterations
        num_eval_sample (int): evaluating belief samples is costly,
            so subsample belief particles
        num_task_sample (int): # task candidates for proposal

    """

    def __init__(
        self,
        env,
        model,
        acquire_fns,
        eval_tasks,
        iterations=10,
        num_eval_sample=5,
        num_task_sample=4,
        fixed_candidates=None,
        debug_belief_task=None,
    ):
        self._env = env
        self._model = model
        self._acquire_fns = acquire_fns
        self._eval_tasks = eval_tasks
        # Random key & function
        self._rng_key = None
        self._random_choice = random_choice
        # Numerics
        self._iterations = iterations
        self._num_eval_sample = num_eval_sample
        self._num_task_sample = num_task_sample
        # Task proposal
        self._fixed_candidates = fixed_candidates
        self._debug_belief_task = debug_belief_task
        # Cache data
        self._acquire_samples = {}
        for key in acquire_fns.keys():
            self._acquire_samples[key] = []

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._model.update_key(rng_key)
        self._random_choice = seed(self._random_choice, rng_key)

    def run(self, task):
        """Run main evaluation loop."""
        curr_tasks, curr_task_names, curr_obs = {}, {}, {}
        for key in self._acquire_fns.keys():
            # Build history for each acquisition function
            curr_obs[key] = []
            curr_tasks[key] = [task]
            curr_task_names[key] = f"ird_{str(task)}"

        for it in range(self._iterations):
            """ Candidates for next tasks """
            if self._fixed_candidates is not None:
                candidates = self._fixed_candidates
            else:
                candidates = self._random_choice(
                    self._env.all_tasks, self._num_task_sample
                )

            """ Run Active IRD on Candidates """
            print(f"\nActive IRD iteration {it}")
            for key in curr_tasks.keys():
                ## Collect observation
                task = curr_tasks[key][-1]
                task_name = curr_task_names[key][-1]
                obs = self._model.simulate_designer(task, task_name)
                curr_obs[key].append(obs)
                print(f"Fn name: {key}; Task name {task_name}")

                ## Main IRD Sampling
                belief = self._model.sample(
                    curr_tasks[key],
                    curr_task_names[key],
                    obs=curr_obs[key],
                    visualize=True,
                )
                ## Evaluate
                if self._debug_belief_task is not None:
                    self._debug_belief(belief, self._debug_belief_task, obs)
                # self._evaluate_violation(belief)
                # self._evaluate_reward(belief)

                ## Actively propose & Record next task
                next_task = self._propose_task(candidates, belief, obs, task, task_name)
                curr_tasks[key].append(next_task)

    def _debug_belief(self, belief, task, obs):
        print(f"Observed weights {obs.weights[0]}")
        task_name = f"debug_{str(task)}"
        feats = belief.get_features(task, task_name)
        violations = belief.get_violations(task, task_name)
        # if np.sum(info["metadata"]["overtake1"]) > 0:
        #    import pdb; pdb.set_trace()

    def _propose_task(self, candidates, belief, obs, curr_task, curr_task_name):
        """Propose next task to designer.

        Args:
            candidates (list): potential next tasks
            obs (Particle[1]): current observed weight

        Note:
            * Require `tasks = env.sample_task()`
            * Use small task space to avoid being too slow.

        TODO:
            * `env.sample_task()`: small number of samples
        """
        acquire_scores = {}
        acquire_tasks = {}
        for key, ac_fn in self._acquire_fns.items():
            acquire_scores[key] = []
            for next_task in candidates:
                next_task_name = f"acquire_{next_task}"
                acquire_scores[key].append(
                    ac_fn(next_task, next_task_name, belief, obs)
                )

        import pdb

        pdb.set_trace()
        """ Find best task for each acquire function """
        for key, scores in acquire_scores.items():
            acquire_tasks[key] = candidates[np.argsort(scores)[0]]

    def _evaluate_violation(self, belief):
        """Evaluate current sampled belief based on violations.

        Note:
            * Require `runner` to collect constraint violations.
            * The fewer violations, the better.

        """
        num_violate = 0.0
        for task in tqdm(self._eval_tasks, desc="Evaluating samples"):
            task_name = f"eval_{task}"
            violations = belief.get_violations(task, task_name)
            # num_violate += sum([sum(v) for v in vio.values()])
        # print(f"Average Violation {num_violate / len(self._eval_tasks):.2f}")

    def _evaluate_reward(self, belief, task, task_name):
        """Evaluate current sampled belief.

        Note:
            * The higher the better.

        """
        for task in tqdm(self._eval_tasks, desc="Evaluating samples"):
            task_name = f"eval_{task}"
            comparison = belief.compare_with(task, task_name, eval_ws)
        print(f"Average Violation {num_violate / len(self._eval_tasks):.2f}")
