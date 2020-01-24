"""Sampling-based probability over reward weights.

"""
from rdb.infer.utils import collect_trajs, random_choice, get_init_state
from rdb.optim.utils import (
    multiply_dict_by_keys,
    subtract_dict_by_keys,
    concate_dict_by_keys,
    divide_dict_by_keys,
)
from rdb.visualize.plot import plot_weights, plot_rankings
from numpyro.handlers import scale, condition, seed
from fast_histogram import histogram1d
from scipy.stats import gaussian_kde
from rdb.exps.utils import Profiler
import jax.numpy as np
import numpy as onp
import copy
import math


class Particles(object):
    """Finite sample set. Used to model belief distribution, normalizer, etc.

    Args:
        sample_ws (list)
        rng_key (jax.random): if None, need to call `particles.update_key()`
        env (object): provide to save time

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller,
        runner,
        sample_ws=None,
        sample_concate_ws=None,
        test_mode=False,
        env=None,
    ):
        self._env_fn = env_fn
        self._env = env
        self._controller = controller
        self._runner = runner
        self._sample_ws = sample_ws
        self._sample_concate_ws = sample_concate_ws
        self._rng_key = rng_key
        self._test_mode = test_mode
        ## Cache data
        self.build_cache()
        ## Sampling function
        if rng_key is None:
            self._random_choice = None
        else:
            self._random_choice = seed(random_choice, rng_key)

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._random_choice = seed(random_choice, rng_key)

    def build_cache(self):
        self._sample_feats = {}
        self._sample_feats_sum = {}
        self._sample_violations = {}
        self._sample_actions = {}

    def update_weights(self, sample_ws=None, sample_concate_ws=None):
        self._sample_ws = sample_ws
        self._sample_concate_ws = sample_concate_ws
        self.build_cache()

    @property
    def rng_key(self):
        return self._rng_key

    @property
    def test_mode(self):
        return self._test_mode

    @test_mode.setter
    def test_mode(self, mode):
        self._test_mode = mode

    @property
    def weights(self):
        if self._sample_ws is None:
            assert (
                self._sample_concate_ws is not None
            ), "Must properly initialize particle weights"
            self._sample_ws = divide_dict_by_keys(self._sample_concate_ws)
        return self._sample_ws

    @weights.setter
    def weights(self, ws):
        self.update_weights(sample_ws=ws)

    @property
    def concate_weights(self):
        if self._sample_concate_ws is None:
            assert (
                self._sample_ws is not None
            ), "Must properly initialize particle weights"
            self._sample_concate_ws = concate_dict_by_keys(self._sample_ws)
        return self._sample_concate_ws

    @property
    def cached_names(self):
        return list(self._sample_feats.keys())

    def subsample(self, num_samples=None):
        """Subsample from current list.

        Usage:
            * Simulate designer.

        """
        assert self._random_choice is not None, "Need to initialize"
        if num_samples is None or num_samples < 0:
            return self
        else:
            assert (
                len(self._sample_ws) >= num_samples
            ), f"Not enough samples for {num_samples}."
            sub_ws = self._random_choice(self._sample_ws, num_samples, replacement=True)
            return Particles(
                self._rng_key,
                self._env_fn,
                self._controller,
                self._runner,
                sub_ws,
                test_mode=self._test_mode,
                env=self._env,
            )

    def log_samples(self, num_samples=None):
        """Print samples to terminal for inspection.
        """
        samples = self.subsample(num_samples)
        for iw, ws in enumerate(samples.weights):
            print(f"Sample {iw}")
            for key, val in ws.items():
                print(f"  {key}: {val:.3f}")

    def get_features(self, task, task_name, desc=None):
        """Compute expected features for sample weights on task.

        Note:
            * Computing feature is costly. Caches features under task_name.

        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_feats[task_name]

    def get_features_sum(self, task, task_name, desc=None):
        """Compute expected feature sums for sample weights on task.
        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_feats_sum[task_name]

    def get_violations(self, task, task_name, desc=None):
        """Compute violations (cached) for sample weights on task."""
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        violations = self._sample_violations[task_name]
        for key, val in violations.items():
            violations[key] = val.sum(axis=-1)
        return violations

    def get_actions(self, task, task_name, desc=None):
        """Compute actions (cached) for sample weights on task."""
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_actions[task_name]

    def _cache_task(self, task, task_name, desc=None):
        if self._env is None:
            self._env = self._env_fn()
        state = get_init_state(self._env, task)
        if self._test_mode:
            dummy_dict = concate_dict_by_keys(self.weights)
            actions, feats, feats_sum, violations = (
                None,
                dummy_dict,
                dummy_dict,
                dummy_dict,
            )
        else:
            actions, feats, feats_sum, violations = collect_trajs(
                self.weights, state, self._controller, self._runner, desc=desc
            )
        self._sample_actions[task_name] = actions
        self._sample_feats[task_name] = feats
        self._sample_feats_sum[task_name] = feats_sum
        self._sample_violations[task_name] = violations

    def dump_task(self, task, task_name):
        """Dump data from current Particles instance.

        Usage:
            >>> particles2.merge(particles1.dump_task(task, task_name))
        """
        return dict(
            task=task,
            task_name=task_name,
            actions=self._sample_actions[task_name],
            feats=self._sample_feats[task_name],
            feats_sum=self._sample_feats_sum[task_name],
            violations=self._sample_violations[task_name],
        )

    def merge(self, data):
        """Merge dumped data from another Particles instance."""
        task_name = data["task_name"]
        if task_name not in self.cached_names:
            self._sample_actions[task_name] = data["actions"]
            self._sample_feats[task_name] = data["feats"]
            self._sample_feats_sum[task_name] = data["feats_sum"]
            self._sample_violations[task_name] = data["violations"]

    def count_violations(self, task, task_name):
        """Roll out features under task.

        Requires:
            * compute_features

        """
        if self._test_mode:
            return 0.0

        if self._env is None:
            self._env = self._env_fn()
        state = get_init_state(self._env, task)
        num_violate = 0.0
        for w in self.weights:
            actions = self._controller(state, weights=w)
            traj, cost, info = self._runner(state, actions, weights=w)
            violations = info["violations"]
            num = sum([sum(v) for v in violations.values()])
            # print(f"violate {num} acs {onp.mean(actions):.3f} xs {onp.mean(traj):.3f}")
            num_violate += num
        return float(num_violate) / len(self.weights)

    def compare_with(self, task, task_name, target, verbose=False):
        """Compare with a set of target weights (usually true weights). Returns log
        prob ratio of reward, measured by target w.

        Args:
            target (Particles): target to compare against; if None, no target

        Requires:
            * compute_features

        """
        if self._test_mode:
            return 0.0

        if target is not None:
            target_w = target.weights[0]

            ## Compare reward difference
            this_feats_sum = self.get_features_sum(task, task_name)
            this_costs = multiply_dict_by_keys(target_w, this_feats_sum)
            target_feats_sum = target.get_features_sum(task, task_name)
            target_cost = multiply_dict_by_keys(target_w, target_feats_sum)

            diff_costs = subtract_dict_by_keys(this_costs, target_cost)
            diff_rews = -1 * onp.sum(list(diff_costs.values()), axis=0)
            ## Compare violation difference
            this_vios = self.get_violations(task, task_name)
            target_vios = target.get_violations(task, task_name)
            diff_vios = subtract_dict_by_keys(this_vios, target_vios)
            diff_vios = onp.sum(list(diff_vios.values()), axis=0)
            if verbose:
                print(
                    f"Diff rew {len(diff_rews)} items: mean {diff_rews.mean():.3f} std {diff_rews.std():.3f} max {diff_rews.max():.3f} min {diff_rews.min():.3f}"
                )
            return diff_rews, diff_vios
        else:
            this_vios = self.get_violations(task, task_name)
            this_vios = onp.sum(list(this_vios.values()), axis=0)
            diff_rews = np.zeros(1)
            return diff_rews, this_vios

    def resample(self, probs):
        """Resample from particles using list of probs. Similar to particle filter update."""
        assert (
            self._random_choice is not None
        ), "Must properly initialize particle weights"
        N = len(self.weights)
        idxs = self._random_choice(onp.arange(N), num=N, probs=probs, replacement=True)
        new_concate_ws = dict()
        for key, value in self.concate_weights.items():
            new_concate_ws[key] = value[idxs]
        new_ps = Particles(
            self._rng_key,
            self._env_fn,
            self._controller,
            self._runner,
            sample_concate_ws=new_concate_ws,
            test_mode=self._test_mode,
        )
        return new_ps

    def entropy(self, method="histogram", bins=50, max_weight=5.0, verbose=True):
        """Estimate entropy.

        Note:
            * Gaussian histogram may cause instability

        TODO:
            * assumes that first weight is unchanging
            * may be sensitive to histogram params (bins, ranges)

        """
        FAST_HISTOGRAM = True

        ranges = (-max_weight, max_weight)
        data = onp.array(list(self.concate_weights.values()))
        # Omit first weight
        data = onp.log(data[1:, :])
        if method == "gaussian":
            # scipy gaussian kde requires transpose
            kernel = gaussian_kde(data)
            N = data.shape[1]
            entropy = -(1.0 / N) * onp.sum(onp.log(kernel(data)))
        elif method == "histogram":
            entropy = 0.0
            for row in data:
                if FAST_HISTOGRAM:
                    hist_count = histogram1d(row, bins=bins, range=ranges)
                    hist_prob = hist_count / len(row)
                    delta = (ranges[1] - ranges[0]) / bins
                    hist_density = hist_prob / delta
                else:
                    # density is normalized by bucket width
                    hist_density, slots = onp.histogram(
                        row, bins=bins, range=ranges, density=True
                    )
                    delta = slots[1] - slots[0]
                    hist_prob = hist_density * delta
                ent = -(hist_density * onp.ma.log(onp.abs(hist_density)) * delta).sum()
                entropy += ent
            # if verbose:
            #    print(f"Entropy {entropy:.3f}")
        return entropy

    def map_estimate(self, num_map=1, method="histogram", bins=50, ranges=(-5.0, 5.0)):
        """Find maximum a posteriori estimate from current samples.

        Note:
            * weights are high dimensional. We approximately estimate
            density by summing bucket counts.
        """
        data = onp.array(list(self.concate_weights.values()))
        # Omit first weight
        data = onp.log(data[1:, :])

        if method == "histogram":
            bins = onp.linspace(*ranges, bins)
            probs = onp.zeros(data.shape[1])
            for row in data:
                which_bins = onp.digitize(row, bins)
                unique_vals, indices, counts = onp.unique(
                    which_bins, return_index=True, return_counts=True
                )
                for val, ct in zip(unique_vals, counts):
                    val_bins_idx = which_bins == val
                    probs[val_bins_idx] += float(ct) / len(row)
            map_idxs = onp.argsort(-1 * probs)[:num_map]
            map_weights = [self.weights[idx] for idx in map_idxs]
            # MAP esimate usually used for evaluation; thread-safe to provide self._env
            map_particles = Particles(
                self._rng_key,
                self._env_fn,
                self._controller,
                self._runner,
                sample_ws=map_weights,
                env=self._env,
            )
            return map_particles
        else:
            raise NotImplementedError

    def diagnose(
        self,
        task,
        task_name,
        target,
        diagnose_N,
        prefix,
        thumbnail=False,
        desc=None,
        video=True,
    ):
        """ Look at top/mid/bottom * diagnose_N particles and their performance. Record videos.

        Args:
            prefix: include directory info e.g. "data/exp_xxx/save_"

        """
        if self._env is None:
            self._env = self._env_fn()
        init_state = get_init_state(self._env, task)

        if thumbnail:
            tbn_path = f"{prefix}_task.png"
            self._runner.collect_thumbnail(init_state, path=tbn_path)

        if video:
            assert len(self.weights) >= diagnose_N * 3
            diff_rews, diff_vios = self.compare_with(task, task_name, target)
            ranks = onp.argsort(diff_rews)
            top_N_idx = ranks[-diagnose_N:]
            mid_N_idx = ranks[
                math.floor((len(ranks) - diagnose_N) / 2) : math.floor(
                    (len(ranks) + diagnose_N) / 2
                )
            ]
            low_N_idx = list(ranks[:diagnose_N])
            top_N_idx, mid_N_idx, low_N_idx = (
                list(top_N_idx),
                list(mid_N_idx),
                list(low_N_idx),
            )
            actions = np.array(self.get_actions(task, task_name))
            top_rews, top_acs = diff_rews[top_N_idx], actions[top_N_idx]
            mid_rews, mid_acs = diff_rews[mid_N_idx], actions[mid_N_idx]
            low_rews, low_acs = diff_rews[low_N_idx], actions[low_N_idx]
            mean_diff = diff_rews.mean()

            map_particles = self.map_estimate(diagnose_N)
            map_acs = np.array(map_particles.get_actions(task, task_name))
            map_rews, map_vios = map_particles.compare_with(task, task_name, target)

            for label, group_rews, group_acs in zip(
                ["map", "top", "mid", "low"],
                [map_rews, top_rews, mid_rews, low_rews],
                [map_acs, top_acs, mid_acs, low_acs],
            ):
                for rew, acs in zip(group_rews, group_acs):
                    file_path = (
                        f"{prefix}label_{label}_mean_{mean_diff:.3f}_rew_{rew:.3f}.mp4"
                    )
                    # Collect proxy reward paths
                    self._runner.collect_mp4(init_state, acs, path=file_path)
                # Collect true reward paths
                target_acs = self._controller(init_state, weights=target.weights[0])
                target_path = f"{prefix}label_{label}_mean_{mean_diff:.3f}_true.mp4"
                self._runner.collect_mp4(init_state, target_acs, path=target_path)

    def record(self, task, task_name, actions, filepath):
        pass

    def visualize(self, path, true_w, obs_w, max_weight=8.0, bins=50):
        """Visualize weight belief distribution in histogram.

        TODO:
            * Slow, takes ~5s for 1000 weight particles

        """
        if self._test_mode:
            return
        map_w = self.map_estimate().weights[0]
        plot_weights(
            self.weights,
            highlight_dicts=[true_w, obs_w, map_w],
            highlight_colors=["r", "k", "m"],
            path=path + ".png",
            title="Proxy Reward; true (red), obs (black) map (magenta)",
            max_weight=max_weight,
        )

    def save(self, path):
        """Save weight belief particles as npz file."""
        # path: ../particles/seed_{}_iteration_{}_method.npz
        with open(path, "wb+") as f:
            np.savez(f, weights=self.weights)

    def load(self, path):
        self._sample_ws = np.load(path, allow_pickle=True)["weights"]

    def visualize_tasks(self, path, suffix, task_names, perfs):
        """Visualize task distribution.

        Args:
            perfs (list): performance, ususally output from self.compare_with
        """
        assert len(self.weights) == 1, "Can only visualize one weight sample"
        vios = concate_dict_by_keys(list(self._sample_violations.values()))
        num_violates = np.array(list(vios.values())).sum(axis=(0, 2, 3))
        # Ranking violations and performance (based on violation)
        plot_rankings(
            num_violates,
            "Violations",
            [perfs],
            ["Perf diff"],
            path=f"{path}_violations{suffix}.png",
            title="Violations",
            yrange=[-20, 20],
        )
        # Ranking performance and violations (based on performance)
        plot_rankings(
            perfs,
            "Perf diff",
            [num_violates],
            ["Violations"],
            path=f"{path}_performance{suffix}.png",
            title="Performance",
            yrange=[-20, 20],
        )
