"""Acquisition Procedures.

Given a new task, compute a saliency score for the new task.

Include:
    * Info Gain: H(X) - H(X | Y)
    * Max discrepancy (ratio): f(X, Y) / f(X)
"""

import jax.numpy as np


class ActiveInfoGain(object):
    """Information Gain.

    Args:
        model (object): IRD Model
    """

    def __init__(self, env, model, num_designers=5, debug=False):
        self._env = env
        self._model = model
        self._num_designers = 5
        self._tasks = []
        self._debug = debug

    def __call__(
        self, next_task, next_task_name, curr_task, curr_task_name, random_choice_fn
    ):
        """
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
    def __init__(self, env, model, method="mean", debug=False):
        super().__init__(env, model, num_designers=-1, debug=debug)
        self._method = method

    def _compute_log_ratios(self, sample_ws, next_feats, user_feats):
        """Compares user features with belief sample features (from samplw_ws).

        TODO:
            * Ugly API, assumes next_feats is 1 designer sample

        """

        log_ratios = []
        assert len(user_feats) == 1
        user_feat = user_feats[0]
        for sample_w, next_feat in zip(sample_ws, next_feats):
            diff_cost = 0
            for key in sample_w.keys():
                # sum over episode
                w = sample_w[key]
                diff_cost += w * np.sum(user_feat[key] - next_feat[key])
            diff_rew = -1 * diff_cost
            log_ratios.append(diff_rew)
        return np.array(log_ratios)

    def __call__(
        self,
        next_task,
        next_task_name,
        curr_task,
        curr_task_name,
        obs_w,
        random_choice_fn,
    ):
        """Disagreement criteria.

        Score = -1 * rew(sample_w, traj_user)/rew(sample_w, traj_sample).

        Note:
            * The higher the better.

        """
        # Get current cached belief samples (TODO: may be error prone)
        curr_ws, curr_feats = self._model.get_samples(curr_task, curr_task_name)
        next_feats = self._model.get_features(next_task, next_task_name, curr_ws)
        user_feats = self._model.get_features(next_task, next_task_name, [obs_w])
        log_ratios = self._compute_log_ratios(curr_ws, next_feats, user_feats)
        if self._method == "mean":
            return -1 * np.mean(log_ratios)
        elif self._method == "min":
            if self._debug:
                min_idx = np.argmin(log_ratios)
                print(f"Min weight {curr_ws[min_idx]}")
            return -1 * np.min(log_ratios)
        else:
            raise NotImplementedError
