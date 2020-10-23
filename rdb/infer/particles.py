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
from functools import partial
from jax import random
import jax.numpy as jnp
import numpy as onp
import copy
import math
import time
import os


class Particles(object):
    """Finite sample set. Used to model belief distribution, normalizer, etc for reward weights design.

    Args:
        weights (DictList)
        rng_key (jax.random): if None, need to call `particles.update_key()`
        env_fn (fn): environment creation function, use `env_fn` instead of `env` to prevent multi-threaded tampering
        env (object): environment object, only pass in if you are sure its safe.
        save_name (str): save file name, e.g. "weights_seed_{self._rng_name}_{name}"
        weight_params (dict): for histogram visualization: MAX_WEIGHT, NUM_BINS, etc

    Note:
        * Supports caching. E.g.
        >>> particles.get_features_sum(tasks) # first time slow
        >>> particles.get_features_sum(tasks) # second time cached

    """

    def __init__(
        self,
        rng_name,
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
        risk_averse=False,
    ):
        self._env_fn = env_fn
        self._env = env
        self._controller = controller
        self._runner = runner

        ## Risk averse property
        self._risk_averse = risk_averse

        ## Sample weights
        self._rng_key = rng_key
        self._rng_name = rng_name
        self._weight_params = weight_params
        self._normalized_key = normalized_key
        if weights is not None:
            # print(type(weights))
            # print(type(weights) == rdb.infer.dictlist.DictList)
            # print(type(weights) == DictList)
            # assert type(weights) == rdb.infer.dictlist.DictList or type(weights) == DictList
            self._weights = weights.normalize_by_key(self._normalized_key)
        else:
            self._weights = None
        self._importance = None  # set plot weights

        self._save_name = save_name
        self._expanded_name = f"weights_seed_{self._rng_name}_{self._save_name}"

        ## File system
        self._fig_dir = fig_dir
        self._save_dir = save_dir
        if fig_dir is not None:
            os.makedirs(fig_dir, exist_ok=True)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        ## Cache data
        self.build_cache()
        self._hessian_computed = False

    def update_key(self, rng_key):
        self._rng_key, rng_choice = random.split(rng_key)
        self._expanded_name = f"weights_seed_{self._rng_name}_{self._save_name}"

    def build_cache(self):
        self._cache_actions = {}
        self._cache_feats = {}
        self._cache_costs = {}
        self._cache_feats_sum = {}
        self._cache_violations = {}
        ## Lowerbound
        self._cache_feats_lb = {}
        self._cache_costs_lb = {}
        self._cache_feats_sum_lb = {}
        self._cache_violations_lb = {}

    def _merge_dict(self, dicta, dictb):
        """Used when merging with another Particles. Keep common key, value pairs."""
        out = {}
        assert isinstance(dicta, dict) and isinstance(dictb, dict)
        for common_key in [k for k in dicta if k in dictb]:
            if isinstance(dicta[common_key], DictList):
                out[common_key] = dicta[common_key].concat(dictb[common_key])
            else:
                out[common_key] = onp.concatenate(
                    [dicta[common_key], dictb[common_key]]
                )
        return out

    def _index_dict(self, dict_, idx):
        """Used when indexing Particles. Index on every task."""
        out = {}
        for key, val in dict_.items():
            out[key] = val[idx]
        return out

    @property
    def rng_name(self):
        return self._rng_name

    @rng_name.setter
    def rng_name(self, name):
        self._rng_name = name

    @property
    def risk_averse(self):
        return self._risk_averse

    @risk_averse.setter
    def risk_averse(self, risk_averse):
        self.build_cache()
        self._risk_averse = risk_averse

    @property
    def rng_key(self):
        return self._rng_key

    @property
    def importance(self):
        return self._importance

    @importance.setter
    def importance(self, importance):
        assert len(importance) == len(self._weights)
        self._importance = importance

    def _clone(self, weights):
        return Particles(
            rng_name=self._rng_name,
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

    def copy(self):
        return self._clone(self._weights)

    def combine(self, ps):
        """Combine with new particles and merge experience (intersection).

        Returns a new Particles object and leaves self untouched.

        """
        assert isinstance(ps, Particles)
        new_weights = self.weights.concat(ps.weights)
        new_ps = self._clone(new_weights)
        ## Merge cache
        new_ps._cache_feats = self._merge_dict(self._cache_feats, ps._cache_feats)
        new_ps._cache_costs = self._merge_dict(self._cache_costs, ps._cache_costs)
        new_ps._cache_feats_sum = self._merge_dict(
            self._cache_feats_sum, ps._cache_feats_sum
        )
        new_ps._cache_violations = self._merge_dict(
            self._cache_violations, ps._cache_violations
        )
        new_ps._cache_actions = self._merge_dict(self._cache_actions, ps._cache_actions)

        ## Lb cache
        new_ps._cache_costs_lb = self._merge_dict(
            self._cache_costs_lb, ps._cache_costs_lb
        )
        new_ps._cache_feats_sum_lb = self._merge_dict(
            self._cache_feats_sum_lb, ps._cache_feats_sum_lb
        )
        new_ps._cache_violations_lb = self._merge_dict(
            self._cache_violations_lb, ps._cache_violations_lb
        )
        return new_ps

    def tile(self, num):
        new_weights = self.weights.tile(num, axis=0)
        new_ps = self._clone(new_weights)
        for key in self.cached_names:
            new_ps._cache_feats[key] = self._cache_feats[key].tile(num, axis=0)
            new_ps._cache_feats_sum[key] = self._cache_feats_sum[key].tile(num, axis=0)
            new_ps._cache_violations[key] = self._cache_violations[key].tile(
                num, axis=0
            )
            new_ps._cache_costs[key] = onp.tile(self._cache_costs[key], num)
            ## Lb cache
            new_ps._cache_feats_lb[key] = self._cache_feats_lb[key].tile(num, axis=0)
            new_ps._cache_feats_sum_lb[key] = self._cache_feats_sum_lb[key].tile(
                num, axis=0
            )
            new_ps._cache_violations_lb[key] = self._cache_violations_lb[key].tile(
                num, axis=0
            )
            new_ps._cache_costs_lb[key] = onp.tile(self._cache_costs_lb[key], num)
            ## Actions
            acs_shape = [1] * len(self._cache_actions[key].shape)
            acs_shape[0] = num
            new_ps._cache_actions[key] = onp.tile(self._cache_actions[key], acs_shape)
        return new_ps

    def repeat(self, num):
        new_weights = self.weights.repeat(num, axis=0)
        new_ps = self._clone(new_weights)
        for key in self.cached_names:
            new_ps._cache_feats[key] = self._cache_feats[key].repeat(num, axis=0)
            new_ps._cache_feats_sum[key] = self._cache_feats_sum[key].repeat(
                num, axis=0
            )
            new_ps._cache_violations[key] = self._cache_violations[key].repeat(
                num, axis=0
            )
            new_ps._cache_costs[key] = onp.repeat(self._cache_costs[key], num, axis=0)
            ## Lb cache
            new_ps._cache_feats_lb[key] = self._cache_feats_lb[key].repeat(num, axis=0)
            new_ps._cache_feats_sum_lb[key] = self._cache_feats_sum_lb[key].repeat(
                num, axis=0
            )
            new_ps._cache_violations_lb[key] = self._cache_violations_lb[key].repeat(
                num, axis=0
            )
            new_ps._cache_costs_lb[key] = onp.repeat(
                self._cache_costs_lb[key], num, axis=0
            )
            ## Actions
            new_ps._cache_actions[key] = onp.repeat(
                self._cache_actions[key], num, axis=0
            )
        return new_ps

    @property
    def weights(self):
        """Access weights.

        Output:
            weights (DictList): particle weights
                shape: nfeats * (nweights, )

        """
        return self._weights

    @weights.setter
    def weights(self, ws):
        assert isinstance(ws, DictList)
        self._weights = ws.normalize_by_key(self._normalized_key)
        self.build_cache()

    @property
    def cached_names(self):
        return list(self._cache_feats.keys())

    def get_task_name(self, task):
        assert (
            len(onp.array(task).shape) == 1
        ), f"Task must be 1D, got shape {onp.array(task).shape}"
        return str(list(task))

    def subsample(self, num_samples=None):
        """Subsample from current list.

        Usage:
            * Simulate designer.

        Output:
            ps (Particles(num_samples))

        """
        if num_samples is None or num_samples < 0:
            return self
        else:
            # assert (
            #     len(self._weights) >= num_samples
            # ), f"Not enough samples for {num_samples}."
            self._rng_key, rng_random = random.split(self._rng_key)
            weights = random_choice(
                rng_random,
                self._weights,
                (num_samples,),
                replace=True,
                p=self._importance,
            )
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

    def get_features(self, tasks, lower=False, desc=None):
        """Compute expected features for sample weights on task.

        Return:
            features (DictList): nfeats * (ntasks, nparticles, T)
            lower (bool): return features for lowerbound trajectories

        Note:
            * Computing feature is costly. Caches features under task_name.

        """
        self.compute_tasks(tasks, desc=desc)
        all_feats = []
        for task in tasks:
            task_name = self.get_task_name(task)
            feats = self._cache_feats[task_name]
            #  shape nfeats * (nparticles, T)
            all_feats.append(feats)
        return DictList(all_feats)

    def get_features_sum(self, tasks, lower=False, desc=None):
        """Compute expected feature sums for sample weights on task.

        Return:
            feats_sum (DictList): nfeats * (ntasks, nparticles)
            lower (bool): return feats sums for lowerbound trajectories

        """
        self.compute_tasks(tasks, desc=desc)
        all_feats_sum = []
        for task in tasks:
            task_name = self.get_task_name(task)
            if lower:
                feats_sum = self._cache_feats_sum_lb[task_name]
            else:
                feats_sum = self._cache_feats_sum[task_name]
            #  shape nfeats * (nparticles)
            all_feats_sum.append(feats_sum)
        return DictList(all_feats_sum)

    def get_offset_by_features(self, feats_dict, task_idx=-1):
        """Compute the feature offsets of current particle, based on step-wise feature counts.

        Args:
            feats_dict (DictList): (nfeats, ntasks, nparticles, T)
        Note:
            * Only select the last task in feats_dict: feats_dict[:, -1]

        Return:
            offset (ndarray): (nparticles,)

        """
        feats_keys = self._env.features_keys
        #  shape (nfeats, ntasks, nparticles, T)
        feats_arr = feats_dict.prepare(feats_keys).numpy_array()
        assert len(feats_arr) == len(feats_keys)

        #  shape (nfeats, 1, T)
        that_feats_arr = feats_arr[:, task_idx]
        T = that_feats_arr.shape[-1]

        #  shape nfeats * (nparticles, T)
        weights = self._weights.prepare(feats_keys).expand_dims(1).repeat(T, axis=1)
        #  shape (nparticles, 1), avg across timestep
        that_rews = (weights.numpy_array() * that_feats_arr).sum(axis=0).mean(axis=1)
        return -1 * that_rews

    def get_hessians(self, tasks, weights=None):
        """Compute Hessian information for features.

        Note:
            - Weird API (for convenience): can use other weights
        """
        if not self._hessian_computed:
            hessian_start = time.time()
            print("First time computing hessian")
        states = self._env.get_init_states(onp.array(tasks))
        T, udim = self._controller.T, self._controller.udim
        batch_us = self.get_actions(tasks).reshape((-1, T, udim))
        if weights is None:
            weights = self.weights
        batch_states, batch_weights = cross_product(
            states, self.weights, onp.array, partial(DictList, jax=jax)
        )
        all_hessians, all_hess_sum = [], []
        for state, us, weights in zip(batch_states, batch_us, batch_weights):
            hess, hess_sum = self._runner.compute_hessian(
                state, us, weights, expand_dims=True
            )
            all_hessians.append(hess)
            all_hess_sum.append(hess_sum)
        if not self._hessian_computed:
            self._hessian_computed = True
            print(
                f"First time computing hessian finished: {(time.time() - hessian_start):.2f}"
            )
        return jnp.array(all_hessians), jnp.array(all_hess_sum)

    def get_costs(self, tasks, lower=False, desc=None):
        """Compute expected feature sums for sample weights on task.

        Return:
            costs (ndarray): (ntasks, nparticles)
            lower (bool): return costs for lowerbound trajectories

        """
        self.compute_tasks(tasks, desc=desc)
        all_costs = []
        for task in tasks:
            task_name = self.get_task_name(task)
            if lower:
                all_costs.append(self._cache_costs_lb[task_name])
            else:
                all_costs.append(self._cache_costs[task_name])
            #  shape nfeats * (nparticles)
        return jnp.array(all_costs)

    def get_violations(self, tasks, lower=False, desc=None):
        """Compute violations sum (cached) for sample weights on task.

        Return:
            violations (DictList): nvios * (ntasks, nparticles)
            lower (bool): return violations for lowerbound trajectories

        """
        self.compute_tasks(tasks, desc=desc)
        all_vios = []
        for task in tasks:
            task_name = self.get_task_name(task)
            if lower:
                #  shape nvios * (nparticles, T)
                violations = self._cache_violations_lb[task_name]
                #  shape nvios * (nparticles)
                vios_sum = violations.sum(axis=1)
            else:
                violations = self._cache_violations[task_name]
                vios_sum = violations.sum(axis=1)
            all_vios.append(vios_sum)
        return DictList(all_vios)

    def get_actions(self, tasks, desc=None):
        """Compute actions (cached) for sample weights on task.

        Return:
            actions (ndarray): (ntasks, nparticles, T, udim)

        """
        self.compute_tasks(tasks, desc=desc)
        acs = [self._cache_actions[self.get_task_name(task)] for task in tasks]
        return onp.array(acs)

    def compute_tasks(
        self, tasks, us0=None, vectorize=True, desc=None, max_batch=500, jax=True
    ):
        """Compute multiple tasks at once.

        Args:
            vectorize (bool): flatten tasks and compute with 1 pass.
            us0 (ndarray): (ntasks, nweights, T, acs_dim)

        Note:
            * vectorize will feed (ntasks * nweights) inputs into controller & runner,
            which may cause extra compile time.

        """
        if self._env is None:
            self._env = self._env_fn()
        ntasks = len(tasks)
        nweights = len(self.weights)
        feats_keys = self._env.features_keys
        T, udim = self._controller.T, self._controller.udim
        new_tasks = [
            task for task in tasks if self.get_task_name(task) not in self.cached_names
        ]
        if vectorize:
            ## Rollout (nweights * ntasks) in one expanded vector
            if len(new_tasks) == 0:
                return
            #  shape (ntasks, state_dim)
            states = self._env.get_init_states(onp.array(tasks))
            if self._risk_averse:
                ## Risk averse planning
                #  - batch_states (ntasks, state_dim)
                #  - batch_weights nfeats * (ntasks, nweights)
                batch_states = states
                batch_weights = self.weights.expand_dims(0).repeat(ntasks, axis=0)
            else:
                ## Batch planning
                #  - batch_states (ntasks * nweights, state_dim)
                #  - batch_weights nfeats * (ntasks * nweights)
                batch_states, batch_weights = cross_product(
                    states, self.weights, onp.array, partial(DictList, jax=jax)
                )
            #  shape (ntasks * nweights, T, udim)
            batch_us0 = None
            if us0 is not None:
                batch_us0 = us0.reshape((-1, T, udim))

            batch_weights_arr = batch_weights.prepare(feats_keys).numpy_array()
            batch_trajs = collect_trajs(
                batch_weights_arr,
                batch_states,
                self._controller,
                self._runner,
                desc=desc,
                us0=batch_us0,
                jax=jax,
                max_batch=max_batch,
            )
            lower_trajs = collect_lowerbound_trajs(
                batch_weights_arr,
                batch_states,
                self._runner,
                jax=jax,
                max_batch=max_batch,
            )
            if self._risk_averse:
                batch_trajs["actions"] = jnp.repeat(
                    jnp.expand_dims(batch_trajs["actions"], 1), nweights, axis=1
                )
                batch_trajs["costs"] = jnp.repeat(
                    jnp.expand_dims(batch_trajs["costs"], 1), nweights, axis=1
                )
                lower_trajs["costs"] = jnp.repeat(
                    jnp.expand_dims(lower_trajs["costs"], 1), nweights, axis=1
                )
                batch_trajs["feats"] = (
                    batch_trajs["feats"].expand_dims(1).repeat(nweights, axis=1)
                )
                lower_trajs["feats"] = (
                    lower_trajs["feats"].expand_dims(1).repeat(nweights, axis=1)
                )
                batch_trajs["feats_sum"] = (
                    batch_trajs["feats_sum"].expand_dims(1).repeat(nweights, axis=1)
                )
                lower_trajs["feats_sum"] = (
                    lower_trajs["feats_sum"].expand_dims(1).repeat(nweights, axis=1)
                )
                batch_trajs["violations"] = (
                    batch_trajs["violations"].expand_dims(1).repeat(nweights, axis=1)
                )
                lower_trajs["violations"] = (
                    lower_trajs["violations"].expand_dims(1).repeat(nweights, axis=1)
                )
            #  shape (ntasks, nweights, T, acs_dim)
            all_actions = batch_trajs["actions"].reshape((ntasks, nweights, T, udim))
            #  shape (ntasks, nweights)
            all_costs = batch_trajs["costs"].reshape((ntasks, nweights))
            low_costs = lower_trajs["costs"].reshape((ntasks, nweights))
            #  shape (ntasks, nweights, T)
            all_feats = batch_trajs["feats"].reshape((ntasks, nweights, T))
            low_feats = lower_trajs["feats"].reshape((ntasks, nweights, T))
            #  shape (ntasks, nweights)
            all_feats_sum = batch_trajs["feats_sum"].reshape((ntasks, nweights))
            low_feats_sum = lower_trajs["feats_sum"].reshape((ntasks, nweights))
            #  shape (ntasks, nweights, T)
            all_vios = batch_trajs["violations"].reshape((ntasks, nweights, T))
            low_vios = lower_trajs["violations"].reshape((ntasks, nweights, T))
        else:
            ## Rollout (nweights,) iteratively for each task in (ntasks,)
            assert not self._risk_averse
            all_actions, all_vios, all_costs = [], [], []
            low_actions, low_vios, low_costs = [], [], []
            all_feats, all_feats_sum = [], []
            low_feats, low_feats_sum = [], []
            batch_us0 = None
            cached_names = self.cached_names
            if us0 is not None:
                #  shape (ntasks * nweights, 1, T, udim)
                batch_us0 = us0.reshape((-1, T, udim)).expand_dims(axis=1)
            else:
                batch_us0 = [None] * ntasks
            weights_arr = self.weights.prepare(feats_keys).numpy_array()
            for ti, task_i in enumerate(tasks):
                name_i = self.get_task_name(task_i)
                if name_i not in cached_names:
                    #  shape (1, task_dim)
                    state_i = self._env.get_init_states([task_i])
                    #  shape (nweights, task_dim)
                    batch_states = jnp.tile(state_i, (nweights, 1))
                    us0_i = batch_us0[ti]
                    trajs = collect_trajs(
                        weights_arr,
                        batch_states,
                        self._controller,
                        self._runner,
                        us0=us0_i,
                        jax=jax,
                        max_batch=max_batch,
                    )
                    low_trajs = collect_lowerbound_trajs(
                        weights_arr,
                        batch_states,
                        self._runner,
                        jax=jax,
                        max_batch=max_batch,
                    )
                    all_actions.append(trajs["actions"])
                    all_vios.append(trajs["violations"])
                    all_feats.append(trajs["feats"])
                    all_costs.append(trajs["costs"])
                    all_feats_sum.append(trajs["feats_sum"])
                    low_vios.append(low_trajs["violations"])
                    low_feats.append(low_trajs["feats"])
                    low_costs.append(low_trajs["costs"])
                    low_feats_sum.append(low_trajs["feats_sum"])
                else:
                    all_actions.append(self._cache_actions[name_i])
                    all_vios.append(self._cache_violations[name_i])
                    all_feats.append(self._cache_feats[name_i])
                    all_costs.append(self._cache_costs[name_i])
                    all_feats_sum.append(self._cache_feats_sum[name_i])
                    low_vios.append(self._cache_violations_lb[name_i])
                    low_feats.append(self._cache_feats_lb[name_i])
                    low_costs.append(self._cache_costs_lb[name_i])
                    low_feats_sum.append(self._cache_feats_sum_lb[name_i])

        ## Cache
        for i, task in enumerate(tasks):
            task_name = self.get_task_name(task)
            self._cache_actions[task_name] = all_actions[i]
            self._cache_feats[task_name] = all_feats[i]
            self._cache_costs[task_name] = all_costs[i]
            self._cache_feats_sum[task_name] = all_feats_sum[i]
            self._cache_violations[task_name] = all_vios[i]
            self._cache_feats_lb[task_name] = low_feats[i]
            self._cache_costs_lb[task_name] = low_costs[i]
            self._cache_feats_sum_lb[task_name] = low_feats_sum[i]
            self._cache_violations_lb[task_name] = low_vios[i]

    def __getitem__(self, key):
        """Indexing by key or by index.
        """
        if (
            isinstance(key, int)
            or isinstance(key, onp.ndarray)
            or isinstance(key, jnp.ndarray)
            or isinstance(key, list)
        ):
            # index
            new_ws = self.weights[key]
            if len(onp.array(key).shape) == 0:
                new_ws = [new_ws]
            new_ws = DictList(new_ws)
            new_ps = self._clone(new_ws)
            new_ps._cache_actions = self._index_dict(self._cache_actions, key)
            new_ps._cache_feats = self._index_dict(self._cache_feats, key)
            new_ps._cache_feats_lb = self._index_dict(self._cache_feats_lb, key)
            new_ps._cache_costs = self._index_dict(self._cache_costs, key)
            new_ps._cache_costs_lb = self._index_dict(self._cache_costs_lb, key)
            new_ps._cache_feats_sum = self._index_dict(self._cache_feats_sum, key)
            new_ps._cache_feats_sum_lb = self._index_dict(self._cache_feats_sum_lb, key)
            new_ps._cache_violations = self._index_dict(self._cache_violations, key)
            new_ps._cache_violations_lb = self._index_dict(
                self._cache_violations_lb, key
            )
            return new_ps

        else:
            raise NotImplementedError
        return val

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        """Iterator to do `for d in dictlist`"""
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def dump_tasks(self, tasks):
        """Dump data from current Particles instance.

        Usage:
            >>> particles2.merge_task(particles1.dump_task(task, task_name))

        """
        out = {}
        for task in tasks:
            task_name = self.get_task_name(task)
            out[task_name] = dict(
                task=task,
                task_name=task_name,
                cache_actions=self._cache_actions[task_name],
                cache_feats=self._cache_feats[task_name],
                cache_feats_lb=self._cache_feats_lb[task_name],
                cache_feats_sum=self._cache_feats_sum[task_name],
                cache_feats_sum_lb=self._cache_feats_sum_lb[task_name],
                cache_costs=self._cache_costs[task_name],
                cache_costs_lb=self._cache_costs_lb[task_name],
                cache_violations=self._cache_violations[task_name],
                cache_violations_lb=self._cache_violations_lb[task_name],
            )
        return out

    def merge_tasks(self, tasks, data):
        """Merge dumped data from another Particles instance.

        Args:
            tasks (ndarray): (ntasks, task_dim)
            data (dict): task_name -> {"actions": (nweights, T, task_dim), ...}

        """
        for task in tasks:
            task_name = self.get_task_name(task)
            assert task_name in data
            if task_name not in self.cached_names:
                dt = data[task_name]
                self._cache_actions[task_name] = dt["cache_actions"]
                self._cache_feats[task_name] = dt["cache_feats"]
                self._cache_feats_lb[task_name] = dt["cache_feats_lb"]
                self._cache_feats_sum[task_name] = dt["cache_feats_sum"]
                self._cache_feats_sum_lb[task_name] = dt["cache_feats_sum_lb"]
                self._cache_costs[task_name] = dt["cache_costs"]
                self._cache_costs_lb[task_name] = dt["cache_costs_lb"]
                self._cache_violations[task_name] = dt["cache_violations"]
                self._cache_violations_lb[task_name] = dt["cache_violations_lb"]

    def merge_bulk_tasks(self, tasks, bulk_data):
        """Merge dumped data from distributed bulk data.

        Args:
            tasks (ndarray): (ntasks, task_dim)
            data (dict): {"actions": (ntasks, nweights, T, task_dim), ...}

        """
        assert len(tasks) == bulk_data["cache_actions"].shape[0]
        assert len(self.weights) == bulk_data["cache_actions"].shape[1]
        for ti, task in enumerate(tasks):
            task_name = self.get_task_name(task)
            if task_name not in self.cached_names:
                self._cache_actions[task_name] = bulk_data["cache_actions"][ti]
                self._cache_costs[task_name] = bulk_data["cache_costs"][ti]
                self._cache_feats[task_name] = bulk_data["cache_feats"][ti]
                self._cache_feats_sum[task_name] = bulk_data["cache_feats_sum"][ti]
                self._cache_violations[task_name] = bulk_data["cache_violations"][ti]
                self._cache_costs_lb[task_name] = bulk_data["cache_costs_lb"][ti]
                self._cache_feats_lb[task_name] = bulk_data["cache_feats_lb"][ti]
                self._cache_feats_sum_lb[task_name] = bulk_data["cache_feats_sum_lb"][
                    ti
                ]
                self._cache_violations_lb[task_name] = bulk_data["cache_violations_lb"][
                    ti
                ]

    def compare_with(
        self, task, target, relative=False, normalized=False, risk_averse=False
    ):
        """Compare with a set of target weights (usually true weights).

        Returns:
            (1) reward (if target w = None)
            (2) regret measured by target w (if target w specified)
            (3) nromalized regret measured by target w (if target w specified, normalized=True)
                relative = (rew - rew_{u_0}) / (rew_{u_target} - rew_{u_0}), where
                u_0 is all zero action, u_target is optimal action under target w

        Note:
            - Relative difference (rew - rew_{u_0}) / (rew_{u_target} - rew_{u_0})
              u_0 is all zero action, relative is not contained between 0~1
            - Normalized difference (rew - rew_min) / (rew_{u_target} - rew_min)
              Contained between 0~1

        Args:
            target (Particles): target to compare against; if None, no target
            relative (bool):
            normalized (bool):

        Output - dict with following keys:
            rews (ndarray): (nbatch,)
            vios (ndarray): (nbatch,)
            vios_by_name (DictList): (nvios, nbatch)

        Requires:
            * compute_features

        """
        eps = 1e-8
        nbatch = len(self.weights)
        if risk_averse:
            nbatch = 1
        feats_keys = self._env.features_keys
        #  shape nfeats * (nbatch, )
        this_fsums = self.get_features_sum([task])[0]
        #  shape nvios * (nbatch,)
        this_vios = self.get_violations([task])[0]
        if target is not None:
            #  shape nfeats * (nbatch, )
            target_ws = target.weights.prepare(feats_keys).tile(nbatch, axis=0)
            target_ws = target_ws.normalize_by_key(self._normalized_key)
            assert len(target.weights) == 1, "Can only compare with 1 target weights."
            that_fsums = target.get_features_sum([task])[0]
            that_vios = target.get_violations([task])[0]
            ## Compute absolute difference
            diff_costs = target_ws * (this_fsums - that_fsums)
            diff_vios_by_name = this_vios - that_vios
            #  shape (nbatch,), average across feats
            diff_rews = -1 * diff_costs.onp_array().mean(axis=0)
            diff_vios_sum = diff_vios_by_name.onp_array().sum(axis=0)

            ## Compute relative difference
            lower_fsums = self.get_features_sum([task], lower=True)[0]
            lower_vios = self.get_violations([task], lower=True)[0]
            max_diff_costs = target_ws * (that_fsums - lower_fsums)
            #  shape (nbatch,), average across feats
            max_diff_rews = -1 * max_diff_costs.onp_array().mean(axis=0)
            max_diff_vios_by_name = that_vios - lower_vios
            max_diff_vios_sum = max_diff_vios_by_name.onp_array().sum(axis=0)

            ## Compute normalized difference
            worst_fsums = DictList([self._env.max_feats_dict])
            worst_diff_costs = target_ws * (that_fsums - worst_fsums)
            #  shape (nbatch,), average across feats
            worst_diff_rews = -1 * worst_diff_costs.onp_array().mean(axis=0)
            return dict(
                rews=diff_rews,
                vios=diff_vios_sum,
                vios_by_name=diff_vios_by_name,
                rews_relative=diff_rews / (max_diff_rews + eps),
                vios_relative=diff_vios_sum / (max_diff_vios_sum + eps),
                rews_normalized=diff_rews / (worst_diff_rews + eps),
            )
        else:
            this_costs = jnp.zeros(nbatch)
            this_rews = jnp.zeros(nbatch)
            this_vios_sum = this_vios.onp_array().sum(axis=0)
            return dict(
                rews=this_rews,
                vios=this_vios_sum,
                vios_by_name=this_vios,
                rews_relative=0.0,
                vios_relative=0.0,
                rews_normalized=0.0,
            )

    def resample(self, probs):
        """Resample from particles using list of new probs. Used for particle filter update."""
        assert len(probs) == len(self.weights)
        self._rng_key, rng_random = random.split(self._rng_key)
        new_weights = random_choice(
            rng_random, self.weights, (len(self.weights),), p=probs, replace=True
        )
        new_weights = DictList(new_weights)
        new_ps = self._clone(new_weights)
        return new_ps

    def entropy(
        self,
        bins,
        max_weights,
        hist_probs=None,
        method="histogram",
        log_scale=False,
        **kwargs,
    ):
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
        delta = (ranges[1] - ranges[0]) / bins
        data = self.weights.copy()
        # Omit normalized weight
        if self._normalized_key in data:
            del data[self._normalized_key]
        #  shape (nfeats - 1, nbatch)
        data = data.onp_array()
        if not log_scale:
            data = onp.log(data)
        if method == "gaussian":
            # scipy gaussian kde requires transpose
            kernel = gaussian_kde(data)
            N = data.shape[1]
            entropy = -(1.0 / N) * onp.sum(onp.log(kernel(data)))
        elif method == "histogram":
            if hist_probs is None:
                hist_probs = self.hist_probs(
                    bins=bins, max_weights=max_weights, log_scale=log_scale
                )
            hist_densities = hist_probs * (1 / delta)
            entropy = 0.0
            for key, density in hist_densities.items():
                ent = -(density * onp.ma.log(onp.abs(density)) * delta).sum()
                entropy += ent
        return entropy

    def hist_probs(self, bins, max_weights, log_scale=False, **kwargs):
        """Histogram probability. How likely does each bin contain samples.

        Return:
            probs (DictList): shape (nfeats - 1) * (nbins,)

        """
        out = {}
        ranges = (-max_weights, max_weights)
        for key in self.weights.keys():
            row = self.weights[key]
            ## Fast histogram
            if not log_scale:
                row = jnp.log(row)
            hist_count = histogram1d(row, bins=bins, range=ranges)
            hist_prob = hist_count / len(row)

            # ## Regular numpy histogram
            # hist_density, slots = onp.histogram(
            #     row, bins=bins, range=ranges, density=True
            # )
            # delta = slots[1] - slots[0]
            # hist_prob = hist_density * delta
            # hist_count = hist_prob * len(row)

            out[key] = hist_prob
        return DictList(out)

    def digitize(self, bins, max_weights, log_scale=False, matrix=False, **kwargs):
        """Based on numpy.digitize. Find histogram bin membership of each sample.

        Args:
            bins: number of bins
            max_weights: weights range [-max_weights, max_weights]
            log_scale: current values are in log_scale
            matrix (bool):
                if true, return (nfeats -1) * nbins
                if false, return (nfeats -1) * (nweights, nbins)

        """
        out = {}
        ranges = (-max_weights, max_weights)
        m_bins = onp.linspace(*ranges, bins)
        nweights = len(self.weights)
        for key in self.weights.keys():
            vals = self.weights[key]
            if not log_scale:
                vals = onp.log(vals)
            which_bins = onp.digitize(vals, m_bins, right=True)
            if not matrix:
                out[key] = which_bins
            else:
                mat = onp.zeros((nweights, bins))
                mat[onp.arange(nweights), which_bins] = 1.0
                out[key] = mat
        return DictList(out)

    def log_prob(self, weights, method="histogram", log_scale=False):
        """Log probability of value"""
        assert (
            "bins" in self._weight_params and "max_weights" in self._weight_params
        ), "Particle weight parameters not properly setup."
        # Normalize by key
        weights = DictList(weights, expand_dims=True).normalize_by_key(
            self._normalized_key
        )[0]
        max_weights = self._weight_params["max_weights"]
        bins = self._weight_params["bins"]
        ranges = (-max_weights, max_weights)
        m_bins = onp.linspace(*ranges, bins)
        log_prob = 0.0
        for key in weights.keys():
            row = self.weights[key]
            val = weights[key]
            if not log_scale:
                row = jnp.log(row)
                val = jnp.log(val)
            unique_vals, indices, counts = onp.unique(
                onp.digitize(row, m_bins, right=True),
                return_index=True,
                return_counts=True,
            )
            # which_bin = onp.digitize(, m_bins, right=True)[0]
            all_counts, _ = onp.histogram(row, bins=bins, range=ranges)
            hist_count, _ = onp.histogram([val], bins=bins, range=ranges)
            ct = onp.sum(hist_count * all_counts)
            prob_ct = float(ct) / len(row)
            eps = 1e-8
            log_prob += jnp.log(prob_ct + eps)
        return log_prob

    def map_estimate(self, num_map, method="histogram", log_scale=False):
        """Find maximum a posteriori estimate from current samples.

        Note:
            * weights are high dimensional. We approximately estimate
            density by summing bucket counts.

        """
        data = self.weights.copy()
        # Omit first weight
        if self._normalized_key in data:
            del data[self._normalized_key]
        #  shape (nfeats - 1, nbatch)
        data = data.onp_array()
        if not log_scale:
            data = onp.log(data)

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
                which_bins = onp.digitize(row, m_bins, right=True)
                unique_vals, indices, counts = onp.unique(
                    which_bins, return_index=True, return_counts=True
                )
                for val, ct in zip(unique_vals, counts):
                    val_bins_idx = which_bins == val
                    prob_ct = float(ct) / len(row)
                    probs[val_bins_idx] += jnp.log(prob_ct)
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
            jnp.savez(f, weights=dict(self.weights))

    def load(self):
        path = f"{self._save_dir}/{self._expanded_name}.npz"
        assert os.path.isfile(path)
        load_data = jnp.load(path, allow_pickle=True)
        self._weights = DictList(load_data["weights"].item())

    def load_from_path(self, path):
        assert os.path.isfile(path)
        load_data = jnp.load(path, allow_pickle=True)
        self._weights = DictList(load_data["weights"].item())

    def visualize(self, true_w=None, obs_w=None, log_scale=False, **kwargs):
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
        map_ws = self.map_estimate(num_map, log_scale=log_scale).weights
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
            bins=bins,
            log_scale=log_scale,
            hist_weights=self._importance,
        )

    def visualize_comparisons(self, tasks, target, fig_name):
        """Visualize comparison with target on multiple tasks.

        Note:
            * A somewhat brittle function for debugging. May be subject to changes

        Args:
            tasks (ndarray): (ntasks, xdim)
            perfs (list): performance, ususally output from self.compare_with

        """
        assert len(self.weights) == 1, "Can only visualize one weight sample"
        diff_rews, diff_vios = [], []
        for task in tasks:
            # (nbatch,)
            comparisons = self.compare_with(task, target)
            diff_rews.append(comparisons["rews"])
            diff_vios.append(comparisons["vios"])
        # Dims: (ntasks, nbatch,) -> (ntasks,)
        diff_rews = onp.array(diff_rews).mean(axis=1)
        diff_vios = onp.array(diff_vios).mean(axis=1)
        # Ranking violations and performance (based on violation)
        prefix = f"{self._fig_dir}/designer_seed_{self._rng_name}"
        plot_rankings(
            main_val=diff_vios,
            main_label="Violations",
            auxiliary_vals=[diff_rews],
            auxiliary_labels=["Perf diff"],
            path=f"{prefix}_violations_{fig_name}.png",
            title="Violations",
            yrange=[-10, 10],
        )
        # Ranking performance and violations (based on performance)
        plot_rankings(
            main_val=diff_rews,
            main_label="Perf diff",
            auxiliary_vals=[diff_vios],
            auxiliary_labels=["Violations"],
            path=f"{prefix}_performance_{fig_name}.png",
            title="Performance",
            yrange=[-10, 10],
        )
