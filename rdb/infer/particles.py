"""Sampling-based probability over reward weights.

"""
from rdb.infer.dictlist import DictList
from rdb.exps.utils import *
from rdb.infer.utils import *
from rdb.optim.utils import *
from rdb.visualize.plot import plot_weights, plot_rankings
from numpyro.handlers import scale, condition, seed
from fast_histogram import histogram1d
from scipy.stats import gaussian_kde
from rdb.exps.utils import Profiler
import jax.numpy as np
import numpy as onp
import copy
import math
import os


class Particles(object):
    """Finite sample set. Used to model belief distribution, normalizer, etc for reward weights design.

    Args:
        weights (DictList)
        rng_key (jax.random): if None, need to call `particles.update_key()`
        env_fn (fn): environment creation function, use `env_fn` instead of `env` to prevent multi-threaded tampering
        env (object): environment object, only pass in if you are sure its safe.
        save_name (str): save file name, e.g. "weights_seed_{str(self._rng_key)}_{name}"
        weight_params (dict): for histogram visualization: MAX_WEIGHT, NUM_BINS, etc

    Note:
        * Supports caching. E.g.
        >>> particles.get_feats(task1, task1_name) # first time slow
        >>> particles.get_feats(task1, task1_name) # second time cached

    """

    def __init__(
        self,
        rng_key,
        env_fn,
        controller,
        runner,
        save_name,
        normalized_key,
        weights=None,
        weight_params={},
        fig_dir=None,
        save_dir=None,
        env=None,
    ):
        self._env_fn = env_fn
        self._env = env
        self._controller = controller
        self._runner = runner

        ## Sample weights
        assert isinstance(weights, DictList)
        self._rng_key = rng_key
        self._weight_params = weight_params
        self._normalized_key = normalized_key
        self._weights = weights.normalize_by_key(self._normalized_key)
        self._expanded_name = f"{save_name}_seed_{str(self._rng_key)}"
        self._save_name = save_name
        ## File system
        self._fig_dir = fig_dir
        self._save_dir = save_dir
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
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
        self._expanded_name = f"weights_seed_{str(self._rng_key)}_{self._save_name}"

    def build_cache(self):
        self._sample_feats = {}
        self._sample_feats_sum = {}
        self._sample_violations = {}
        self._sample_actions = {}

    @property
    def rng_key(self):
        return self._rng_key

    def _clone(self, weights):
        return Particles(
            rng_key=self._rng_key,
            env_fn=self._env_fn,
            controller=self._controller,
            runner=self._runner,
            save_name=self._save_name,
            weights=weights,
            weight_params=self._weight_params,
            normalized_key=self._normalized_key,
            env=self._env,
            fig_dir=self._fig_dir,
            save_dir=self._save_dir,
        )

    def add_weights(self, weights):
        """Add more weight samples.
        """
        assert isinstance(weights, DictList)
        new_weights = self.weights.concat(weights)
        return self._clone(new_weights)

    def tile_weights(self, num):
        new_weights = self.weights.tile(num, axis=0)
        return self._clone(new_weights)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, ws):
        assert isinstance(ws, DictList)
        self._weights = ws.normalize_by_key(self._normalized_key)
        self.build_cache()

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
                len(self._weights) >= num_samples
            ), f"Not enough samples for {num_samples}."
            weights = self._random_choice(self._weights, num_samples, replacement=True)
            weights = DictList(weights)
            return self._clone(weights)

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

        Return:
            features (DictList): (nfeats, nparticles)

        Note:
            * Computing feature is costly. Caches features under task_name.

        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_feats[task_name]

    def get_features_sum(self, task, task_name, desc=None):
        """Compute expected feature sums for sample weights on task.

        Return:
            feats_sum (DictList): (nfeats, nparticles)

        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_feats_sum[task_name]

    def get_violations(self, task, task_name, desc=None):
        """Compute violations sum (cached) for sample weights on task.

        Return:
            violations (DictList): (nvios, nparticles)

        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        violations = self._sample_violations[task_name]
        vios_sum = violations.sum(axis=1)
        return vios_sum

    def get_actions(self, task, task_name, desc=None):
        """Compute actions (cached) for sample weights on task.

        Return:
            actions (ndarray): (nparticles, T, udim)

        """
        if task_name not in self.cached_names:
            self._cache_task(task, task_name, desc)
        return self._sample_actions[task_name]

    def _cache_task(self, task, task_name, desc=None):
        if self._env is None:
            self._env = self._env_fn()
        state = self._env.get_init_state(task)
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

    def compare_with(self, task, task_name, target, verbose=False):
        """Compare with a set of target weights (usually true weights). Returns log
        prob ratio of reward, measured by target w.

        Args:
            target (Particles): target to compare against; if None, no target

        Output:
            diff_rews (ndarray): (nbatch,)
            diff_vios (ndarray): (nbatch,)

        Requires:
            * compute_features

        """
        nbatch = len(self.weights)
        if target is not None:
            # shape (nfeats, nbatch, )
            target_ws = target.weights.tile(nbatch, axis=0)
            target_ws = target_ws.normalize_by_key(self._normalized_key)
            assert len(target.weights) == 1, "Can only compare with 1 target weights."
            ## Compare reward difference
            #  shape (nfeats, nbatch, )
            this_fsums = self.get_features_sum(task, task_name)
            that_fsums = target.get_features_sum(task, task_name)
            #  shape (nfeats, nbatch, )
            diff_costs = target_ws * (this_fsums - that_fsums)
            #  shape (nbatch, )
            diff_rews = -1 * diff_costs.onp_array().sum(axis=0)
            ## Compare violation difference
            #  shape (nvios, nbatch)
            this_vios = self.get_violations(task, task_name)
            that_vios = target.get_violations(task, task_name)
            diff_vios = this_vios - that_vios
            #  shape (nbatch)
            diff_vios = diff_vios.onp_array().sum(axis=0)
            if verbose:
                print(
                    f"Diff rew {len(diff_rews)} items: mean {diff_rews.mean():.3f} std {diff_rews.std():.3f} max {diff_rews.max():.3f} min {diff_rews.min():.3f}"
                )
            return diff_rews, diff_vios
        else:
            this_ws = self.weights
            this_fsums = self.get_features_sum(task, task_name)
            this_costs = this_ws * this_fsums
            this_rews = -1 * this_costs.onp_array().sum(axis=0)
            this_vios = self.get_violations(task, task_name)
            this_vios = this_vios.onp_array().sum(axis=0)
            return this_rews, this_vios

    def resample(self, probs):
        """Resample from particles using list of new probs. Used for particle filter update."""
        assert (
            self._random_choice is not None
        ), "Must properly initialize particle weights"
        assert len(probs) == len(self.weights)
        new_weights = self._random_choice(
            self.weights, num=len(self.weights), probs=probs, replacement=True
        )
        new_weights = DictList(new_weights)
        new_ps = self._clone(new_weights)
        return new_ps

    def entropy(self, bins, max_weights, verbose=True, method="histogram"):
        """Estimate entropy.

        Args:
            bins (int): number of bins
            max_weights (float): log range of weights ~ (-max_weights, max_weights)

        Note:
            * Gaussian histogram may cause instability

        TODO:
            * assumes that first weight is unchanging
            * may be sensitive to histogram params (bins, ranges)

        """
        FAST_HISTOGRAM = True

        ranges = (-max_weights, max_weights)
        data = self.weights.copy()
        # Omit normalized weight
        del data[self._normalized_key]
        #  shape (nfeats - 1, nbatch)
        data = onp.log(data.onp_array())
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
        return entropy

    def map_estimate(self, num_map, method="histogram"):
        """Find maximum a posteriori estimate from current samples.

        Note:
            * weights are high dimensional. We approximately estimate
            density by summing bucket counts.

        """
        data = self.weights.copy()
        # Omit first weight
        del data[self._normalized_key]
        #  shape (nfeats - 1, nbatch)
        data = onp.log(data.onp_array())

        if method == "histogram":
            assert (
                "bins" in self._weight_params and "max_weights" in self._weight_params
            ), "Particle weight parameters not properly setup."
            max_weights = self._weight_params["max_weights"]
            bins = self._weight_params["bins"]
            ranges = (-max_weights, max_weights)
            m_bins = onp.linspace(*ranges, bins)
            probs = onp.zeros(data.shape[1])
            for row in data:
                which_bins = onp.digitize(row, m_bins)
                unique_vals, indices, counts = onp.unique(
                    which_bins, return_index=True, return_counts=True
                )
                for val, ct in zip(unique_vals, counts):
                    val_bins_idx = which_bins == val
                    probs[val_bins_idx] += float(ct) / len(row)
            map_idxs = onp.argsort(-1 * probs)[:num_map]
            map_weights = self.weights[map_idxs]
            # MAP esimate usually used for evaluation; thread-safe to provide self._env
            map_particles = self._clone(map_weights)
            return map_particles
        else:
            raise NotImplementedError

    def save(self):
        """Save weight belief particles as npz file."""
        assert self._save_dir is not None, "Need to specify save directory."
        path = f"{self._save_dir}/{self._expanded_name}.npz"
        with open(path, "wb+") as f:
            np.savez(f, weights=self.weights)

    def load(self):
        path = f"{self._save_dir}/{self._expanded_name}.npz"
        assert os.path.isfile(path)
        load_data = np.load(path, allow_pickle=True)
        self._weights = DictList(load_data["weights"].item())

    def visualize(self, true_w=None, obs_w=None):
        """Visualize weight belief distribution in histogram.

        Shows all weight particles, true weight, observed weight and MAP weights.

        Args:
            true_w (dict): true weight. Specific to IRD experiments.
            obs_w (dict): observed weight (e.g. from designer). Specific to IRD experiments.

        """
        assert self._fig_dir is not None, "Need to specify figure directory."
        bins = self._weight_params["bins"]
        max_weights = self._weight_params["max_weights"]

        num_map = 4  # Magic number for now
        map_ws = self.map_estimate(num_map).weights
        map_ls = [str(i) for i in range(1, 1 + num_map)]
        # Visualize multiple map weights in magenta with ranking labels
        plot_weights(
            self.weights,
            highlight_dicts=[true_w, obs_w] + list(map_ws),
            highlight_colors=["r", "k"] + ["m"] * num_map,
            highlight_labels=["l", "o"] + map_ls,
            path=f"{self._fig_dir}/{self._expanded_name}.png",
            title="Proxy Reward; true (red), obs (black) map (magenta)",
            max_weights=max_weights,
        )

    def visualize_comparisons(self, tasks, task_names, target, fig_name):
        """Visualize comparison with target on multiple tasks.

        Note:
            * A somewhat brittle function for debugging. May be subject to changes

        Args:
            tasks (ndarray): (ntasks, xdim)
            task_names (list)
            perfs (list): performance, ususally output from self.compare_with

        """
        assert len(self.weights) == 1, "Can only visualize one weight sample"
        diff_rews, diff_vios = [], []
        for task, task_name in zip(tasks, task_names):
            # (nbatch,)
            diff_rew, diff_vio = self.compare_with(task, task_name, target)
            diff_rews.append(diff_rew)
            diff_vios.append(diff_vio)
        # Dims: (n_tasks, nbatch,) -> (n_tasks,)
        diff_rews = onp.array(diff_rews).mean(axis=1)
        diff_vios = onp.array(diff_vios).mean(axis=1)
        # Ranking violations and performance (based on violation)
        prefix = f"{self._fig_dir}/designer_seed_{str(self._rng_key)}"
        plot_rankings(
            main_val=diff_vios,
            main_label="Violations",
            auxiliary_vals=[diff_rews],
            auxiliary_labels=["Perf diff"],
            path=f"{prefix}_violations_{fig_name}.png",
            title="Violations",
            yrange=[-20, 20],
        )
        # Ranking performance and violations (based on performance)
        plot_rankings(
            main_val=diff_rews,
            main_label="Perf diff",
            auxiliary_vals=[diff_vios],
            auxiliary_labels=["Violations"],
            path=f"{prefix}_performance_{fig_name}.png",
            title="Performance",
            yrange=[-20, 20],
        )

    # def diagnose(
    #     self,
    #     task,
    #     task_name,
    #     target,
    #     diagnose_N,
    #     prefix,
    #     thumbnail=False,
    #     desc=None,
    #     video=True,
    # ):
    #     """Look at top/mid/bottom * diagnose_N particles and their performance. Record videos.

    #     Note:
    #         * A somewhat brittle function for debugging. May be subject to changes

    #     Args:
    #         prefix: include directory info e.g. "data/exp_xxx/save_"
    #         params (dict) : experiment parameters

    #     """
    #     if self._env is None:
    #         self._env = self._env_fn()
    #     init_state = self._env.get_init_state(task)

    #     if thumbnail:
    #         tbn_path = f"{prefix}_task.png"
    #         self._runner.collect_thumbnail(init_state, path=tbn_path)

    #     if video:
    #         assert len(self.weights) >= diagnose_N * 3
    #         diff_rews, diff_vios = self.compare_with(task, task_name, target)
    #         ranks = onp.argsort(diff_rews)
    #         top_N_idx = ranks[-diagnose_N:]
    #         mid_N_idx = ranks[
    #             math.floor((len(ranks) - diagnose_N) / 2) : math.floor(
    #                 (len(ranks) + diagnose_N) / 2
    #             )
    #         ]
    #         low_N_idx = list(ranks[:diagnose_N])
    #         top_N_idx, mid_N_idx, low_N_idx = (
    #             list(top_N_idx),
    #             list(mid_N_idx),
    #             list(low_N_idx),
    #         )
    #         actions = np.array(self.get_actions(task, task_name))
    #         top_rews, top_acs = diff_rews[top_N_idx], actions[top_N_idx]
    #         mid_rews, mid_acs = diff_rews[mid_N_idx], actions[mid_N_idx]
    #         low_rews, low_acs = diff_rews[low_N_idx], actions[low_N_idx]
    #         mean_diff = diff_rews.mean()

    #         map_particles = self.map_estimate(diagnose_N)
    #         map_acs = np.array(map_particles.get_actions(task, task_name))
    #         map_rews, map_vios = map_particles.compare_with(task, task_name, target)

    #         for label, group_rews, group_acs in zip(
    #             ["map", "top", "mid", "low"],
    #             [map_rews, top_rews, mid_rews, low_rews],
    #             [map_acs, top_acs, mid_acs, low_acs],
    #         ):
    #             for rew, acs in zip(group_rews, group_acs):
    #                 file_path = (
    #                     f"{prefix}label_{label}_mean_{mean_diff:.3f}_rew_{rew:.3f}.mp4"
    #                 )
    #                 # Collect proxy reward paths
    #                 self._runner.collect_mp4(init_state, acs, path=file_path)
    #             # Collect true reward paths
    #             target_acs = self._controller(init_state, weights=target.weights[0])
    #             target_path = f"{prefix}label_{label}_mean_{mean_diff:.3f}_true.mp4"
    #             self._runner.collect_mp4(init_state, target_acs, path=target_path)
