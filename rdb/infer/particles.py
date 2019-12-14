"""Sampling-based probability over reward weights.

"""
from rdb.infer.utils import collect_trajs, random_choice
from numpyro.handlers import scale, condition, seed


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
                len(self._sample_ws) > num_samples
            ), f"Not enough samples for {num_samples}."
            sub_ws = self._random_choice(self._sample_ws, num_samples)
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
        pass

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    def resample(self, probs):
        """Similar to particle filter update, resample from list of existing particles
        using provided list of probabilities.

        """
        raise NotImplementedError
        # Clear cache
        self._sample_feats = {}
        self._sample_feats_sum = {}
        self._sample_violations = {}

    def entropy(self):
        raise NotImplementedError
        pass
