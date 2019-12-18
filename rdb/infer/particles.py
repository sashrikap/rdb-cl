"""Sampling-based probability over reward weights.

"""
from rdb.infer.utils import collect_trajs, random_choice
from rdb.optim.utils import (
    multiply_dict_by_keys,
    subtract_dict_by_keys,
    concate_dict_by_keys,
)
from numpyro.handlers import scale, condition, seed
from scipy.stats import gaussian_kde
import jax.numpy as np
import numpy as onp


class Particles(object):
    """Finite sample set. Used to model belief distribution, normalizer, etc.

    Args:
        sample_ws (list)

    """

    def __init__(self, rng_key, env, controller, runner, sample_ws):
        self._env = env
        self._controller = controller
        self._runner = runner
        self._sample_ws = sample_ws
        self._rng_key = rng_key
        ## Cache data
        self._sample_feats = {}
        self._sample_feats_sum = {}
        self._sample_violations = {}
        self._sample_actions = {}
        ## Sampling function
        self._random_choice = seed(random_choice, rng_key)

    @property
    def weights(self):
        return self._sample_ws

    @property
    def cached_tasks(self):
        return self._sample_feats.keys()

    def subsample(self, num_samples=None):
        """Subsample from current list.

        Usage:
            * Simulate designer.

        """
        if num_samples is None:
            return self
        else:
            assert (
                len(self._sample_ws) >= num_samples
            ), f"Not enough samples for {num_samples}."
            sub_ws = self._random_choice(self._sample_ws, num_samples, replacement=True)
            return Particles(
                self._rng_key, self._env, self._controller, self._runner, sub_ws
            )

    def get_features(self, task, task_name, desc=None):
        """Compute expected features for sample weights on task.

        Note:
            * Computing feature is costly. Caches features under task_name.

        """
        if task_name not in self._sample_feats.keys():
            self._cache_task(task, task_name, desc)
        return self._sample_feats[task_name]

    def get_features_sum(self, task, task_name, desc=None):
        """Compute expected feature sums for sample weights on task.
        """
        if task_name not in self._sample_feats_sum.keys():
            self._cache_task(task, task_name, desc)
        return self._sample_feats_sum[task_name]

    def get_violations(self, task, task_name, desc=None):
        """Compute violations (cached) for sample weights on task."""
        if task_name not in self._sample_violations.keys():
            self._cache_task(task, task_name, desc)
        return self._sample_violations[task_name]

    def get_actions(self, task, task_name, desc=None):
        """Compute actions (cached) for sample weights on task."""
        if task_name not in self._sample_actions.keys():
            self._cache_task(task, task_name, desc)
        return self._sample_actions[task_name]

    def _cache_task(self, task, task_name, desc=None):
        state = self._get_init_state(task)
        actions, feats, feats_sum, violations = collect_trajs(
            self.weights, state, self._controller, self._runner, desc=desc
        )
        self._sample_actions[task_name] = actions
        self._sample_feats[task_name] = feats
        self._sample_feats_sum[task_name] = feats_sum
        self._sample_violations[task_name] = violations

    def count_violations(self, task, task_name):
        """Roll out features under task.

        Requires:
            * compute_features

        """
        state = self._get_init_state(task)
        num_violate = 0.0
        for w in self.weights:
            actions = self._controller(state, weights=w)
            traj, cost, info = self._runner(state, actions, weights=w)
            violations = info["violations"]
            num = sum([sum(v) for v in violations.values()])
            # print(f"violate {num} acs {np.mean(actions):.3f} xs {np.mean(traj):.3f}")
            num_violate += num
        return float(num_violate) / len(self.weights)

    def compare_with(self, task, task_name, target_w):
        """Compare with a set of target weights (usually true weights). Returns log
        prob ratio of reward, measured by target w.

        Requires:
            * compute_features

        """
        this_feats_sum = self.get_features_sum(task, task_name)
        this_cost = multiply_dict_by_keys(target_w, this_feats_sum)

        target = Particles(
            self._rng_key, self._env, self._controller, self._runner, [target_w]
        )
        target_feats_sum = target.get_features_sum(task, task_name)
        target_cost = multiply_dict_by_keys(target_w, target_feats_sum)

        diff_cost = subtract_dict_by_keys(this_cost, target_cost)
        diff_rew = -1 * np.sum(list(diff_cost.values()), axis=0).mean()
        # if diff_rew > 0:
        #    import pdb; pdb.set_trace()
        return diff_rew

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    def resample(self, probs):
        """Resample from particles using list of probs. Similar to particle filter update."""
        new_ws = self._random_choice(
            self.weights, len(self.weights), probs=probs, replacement=True
        )
        return Particles(
            self._rng_key, self._env, self._controller, self._runner, new_ws
        )

    def entropy(self, method="histogram", bins=100, ranges=(-8.0, 8.0)):
        """Estimate entropy

        Note:
            * Gaussian histogram may cause instability

        TODO:
            * assumes that first weight is unchanging

        """

        concate_weights = concate_dict_by_keys(self.weights)
        data = np.array(list(concate_weights.values()))

        # Omit first weight
        data = np.log(data[1:, :])
        if method == "gaussian":
            # scipy gaussian kde requires transpose
            kernel = gaussian_kde(data)
            N = data.shape[1]
            entropy = -(1.0 / N) * np.sum(np.log(kernel(data)))
        elif method == "histogram":
            entropy = 0.0
            for row in data:
                hist = onp.histogram(row, bins=bins, range=ranges, density=True)
                data = hist[0]
                ent = -(data * onp.ma.log(onp.abs(data))).sum()
                entropy += ent
        return entropy

    def visualize(self, path):
        # TODO
        # path ../plots/seed_{}_iteration_{}_method.png
        pass

    def save(self, path):
        # TODO
        # path: ../particles/seed_{}_iteration_{}_method.npz
        pass
