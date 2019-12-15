"""Acquisition Procedures.

Given a new task, compute a saliency score for the new task.

Include:
    * Info Gain: H(X) - H(X | Y)
    * Max discrepancy (ratio): f(X, Y) / f(X)
"""

import jax.numpy as np
from rdb.optim.utils import (
    multiply_dict_by_keys,
    subtract_dict_by_keys,
    concate_dict_by_keys,
)


class ActiveInfoGain(object):
    """Information Gain.

    Args:
        model (object): IRD Model
    """

    def __init__(self, env, model, num_designers=5, num_acquire_sample=5, debug=False):
        self._env = env
        self._model = model
        self._num_designers = 5
        self._num_acquire_sample = num_acquire_sample
        self._tasks = []
        self._debug = debug

    def __call__(self, next_task, next_task_name, belief, obs):
        """Information gain (negative entropy) criteria.

        Note:
            * Equivalent to ranking by negative post-obs entropy -H(X|Y)
              [H(X) - H(X|Y)] - [H(X) - H(X|Y')] = H(X|Y') - H(X|Y)

        Pseudocode:
        ```
        self._tasks.append(task)
        curr_sample_ws, curr_sample_feats = self._model.get_samples(curr_task_name)
        user_w, user_feat = random.choice(curr_sample_ws, curr_sample_feats)
        # Collect featurs on new task
        task_feats = self._model.collect_feats(curr_sample_ws, task)
        for feats in task_feats:
            new_log_prob = user_w.dot(feats)
        new_sample_ws = resample(curr_sample_ws, new_log_prob + log_prob)
        return entropy(curr_sample_ws) - entropy(new_sample_ws)
        ```
        """
        curr_ws, curr_feats = self._model.get_samples(next_task, next_task_name)
        # sample one user
        user_ws, user_feats = random_choice_fn(zip(curr_ws, curr_feats), 1)
        task_feats = self._model.collect_feats


class ActiveRatioTest(ActiveInfoGain):
    """Using performance ratio for acquisition.

    Args:
        num_acquire_sample (int): running acquisition function on belief
            samples is costly, so subsample belief particles

    """

    def __init__(self, env, model, method="mean", num_acquire_sample=5, debug=False):
        super().__init__(
            env,
            model,
            num_acquire_sample=num_acquire_sample,
            num_designers=-1,
            debug=debug,
        )
        self._method = method

    def _compute_log_ratios(self, weights, next_feats_sum, user_feats_sum):
        """Compares user features with belief sample features (from samplw_ws).

        Ratio = exp(next_w @ user_feats) / exp(next_w @ next_feats)

        TODO:
            * Ugly API, assumes next_feats is 1 designer sample

        """

        next_costs = multiply_dict_by_keys(weights, next_feats_sum)
        user_costs = multiply_dict_by_keys(weights, user_feats_sum)
        diff_costs = subtract_dict_by_keys(user_costs, next_costs)
        log_ratios = -1 * np.sum(list(diff_costs.values()), axis=0)
        return np.array(log_ratios)

    def __call__(self, next_task, next_task_name, belief, obs):
        """Disagreement criteria.

        Score = -1 * rew(sample_w, traj_user)/rew(sample_w, traj_sample).

        Note:
            * The higher the better.

        """
        belief = belief.subsample(self._num_acquire_sample)

        next_feats_sum = belief.get_features_sum(
            next_task, next_task_name, f"Computing {self._method} acquisition features"
        )
        user_feats_sum = obs.get_features_sum(next_task, next_task_name)
        weights = concate_dict_by_keys(belief.weights)
        log_ratios = self._compute_log_ratios(weights, next_feats_sum, user_feats_sum)

        if self._method == "mean":
            return -1 * np.mean(log_ratios)
        elif self._method == "min":
            if self._debug:
                min_idx = np.argmin(log_ratios)
                import pdb

                pdb.set_trace()
                print(f"Min weight {belief.weights[min_idx]}")
                print(
                    f"ratios mean {np.mean(log_ratios):.3f} std {np.std(log_ratios):.3f}"
                )
            return -1 * np.min(log_ratios)
        else:
            raise NotImplementedError
