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

    def __init__(self, env, model, num_designers=5, num_acquire_sample=5, debug=False):
        self._env = env
        self._model = model
        self._num_designers = 5
        self._num_acquire_sample = num_acquire_sample
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

        TODO:
            * Ugly API, assumes next_feats is 1 designer sample

        """

        log_ratios = []
        for w_i, ws in enumerate(weights):
            diff_rew = (
                -1
                * np.array(
                    [
                        ws[key] * (user_feats_sum[key][0] - next_feats_sum[key][w_i])
                        for key in ws.keys()
                    ]
                ).sum()
            )
            log_ratios.append(diff_rew)
        return np.array(log_ratios)

    def __call__(self, next_task, next_task_name, belief, obs):
        """Disagreement criteria.

        Score = -1 * rew(sample_w, traj_user)/rew(sample_w, traj_sample).

        Note:
            * The higher the better.

        """
        # Get current cached belief samples (TODO: may be error prone)
        # curr_ws, curr_feats = self._model.get_samples(curr_task, curr_task_name)
        belief = belief.subsample(self._num_acquire_sample)

        next_feats_sum = belief.get_features_sum(next_task, next_task_name)
        user_feats_sum = obs.get_features_sum(next_task, next_task_name)
        log_ratios = self._compute_log_ratios(
            belief.weights, next_feats_sum, user_feats_sum
        )
        if self._method == "mean":
            return -1 * np.mean(log_ratios)
        elif self._method == "min":
            if self._debug:
                min_idx = np.argmin(log_ratios)
                print(f"Min weight {belief.weights[min_idx]}")
                print(
                    f"ratios mean {np.mean(log_ratios):.3f} std {np.std(log_ratios):.3f}"
                )
            import pdb

            pdb.set_trace()
            return -1 * np.min(log_ratios)
        else:
            raise NotImplementedError
