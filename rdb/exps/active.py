"""Acquisition Procedures for ExperimentActiveIRD.

Given a new task, compute a saliency score for the new task.

Include:
    * Info Gain: H(X) - H(X | Y)
    * Max discrepancy (ratio): f(X, Y) / f(X)

Credits:
    * Jerry Z. He 2019-2020

"""

import jax.numpy as np
import numpy as onp
import time
import math
import jax
from rdb.infer.utils import random_uniform
from jax.scipy.special import logsumexp
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from rdb.optim.utils import *
from functools import partial
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random


class ActiveInfoGain(object):
    """Information Gain.

    Args:
        weight_params (dict): parameters (e.g. histogram) for calling.

    """

    def __init__(self, rng_key, beta, weight_params={}, debug=False):
        self._rng_key = rng_key
        self._beta = beta
        self._tasks = []
        self._debug = debug
        self._method = "InfoGain"
        self._weight_params = weight_params
        self._time = None
        self._entropy_fn = self._build_entropy()

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def _build_entropy(self):
        @jax.jit
        def _fn(weights_arr, next_feats_sum, which_bins):
            """
            Args:
                weights_arr (ndarray): shape (nfeats, 1)
                next_feats_sum (ndarray): shape (nfeats, nweights)
                which_bins (ndarray): shape (nfeats, nweights, nbins)

            """
            #  shape (nweights,)
            # import pdb; pdb.set_trace()
            new_costs = (weights_arr * next_feats_sum).mean(axis=0)
            #  shape (nweights,)
            new_rews = -1 * self._beta * new_costs
            #  shape (nweights, 1)
            ratio = np.exp(new_rews - logsumexp(new_rews))[:, None]
            max_weights = self._weight_params["max_weights"]
            bins = self._weight_params["bins"]
            delta = 2 * max_weights / bins

            #  shape (nfeats, nbins)
            next_probs = (which_bins * ratio).sum(axis=1)
            # Normalize
            next_probs = next_probs / next_probs.sum(axis=1, keepdims=True)
            next_densities = next_probs * (1 / delta)
            ent = 0.0
            for density in next_densities:
                # assert np.all(density >= 0)
                abs_nonzero = density > 0
                # Trick by setting 0-density to 1, which passes np.log
                density += 1 - abs_nonzero
                # (density * onp.ma.log(abs_density) * delta).sum()
                ent += -(density * np.log(density) * delta).sum()
            return ent

        return _fn

    def _batch_compute_entropy(self, entropy_vfn, all_weights_arr, batch_size=100):
        """If we compute entropy of all weights in one go, JAX would run out of RAM.

        Args:
            all_weights_arr (ndarray): weight array
                shape (nfeats, nweights,)

        """
        if batch_size < 0:
            return entropy_fvn(all_weights_arr)
        else:
            nbatches = math.ceil(len(all_weights_arr) / batch_size)
            entropies = []
            for bi in range(nbatches):
                idx_start, idx_end = bi * batch_size, (bi + 1) * batch_size
                next_ents = entropy_vfn(all_weights_arr[:, idx_start:idx_end])
                entropies.append(next_ents)
            return np.concatenate(entropies, axis=0)

    def __call__(self, next_task, belief, all_obs, feats_keys, verbose=True):
        """Information gain (negative entropy) criteria. Higher score the better.

        Note:
            * Equivalent to ranking by negative post-obs entropy -H(X|Y)
              [H(X) - H(X|Y)] - [H(X) - H(X|Y')] = H(X|Y') - H(X|Y)
            * Entropy is estimated from histogram

        """
        desc = f"Computing {self._method} acquisition features"
        if not verbose:
            desc = None

        ## Histogram identity
        #  shape (nfeats, nweights, nbins)
        which_bins = belief.digitize(
            log_scale=False, matrix=True, **self._weight_params
        ).numpy_array()
        #  shape (nfeats, nweights)
        next_feats_sum = belief.get_features_sum(next_task, desc=desc).prepare(
            feats_keys
        )
        next_feats_sum = next_feats_sum.squeeze(0).numpy_array()
        #  shape (nfeats, nbins,)
        curr_probs = which_bins.sum(axis=1)
        curr_probs = curr_probs / curr_probs.sum(axis=1, keepdims=True)
        #  shape (nfeats, nweights, 1)
        all_weights_arr = belief.weights.prepare(feats_keys).normalize_across_keys()
        all_weights_arr = all_weights_arr.numpy_array()[:, :, None]

        entropy_vfn = jax.vmap(
            partial(
                self._entropy_fn, next_feats_sum=next_feats_sum, which_bins=which_bins
            ),
            in_axes=1,
        )
        self._entropy_fn(
            all_weights_arr[:, 0], next_feats_sum=next_feats_sum, which_bins=which_bins
        )
        entropies = self._batch_compute_entropy(entropy_vfn, all_weights_arr)
        infogain = -1 * np.mean(entropies)
        assert not np.isnan(infogain)

        if self._debug:
            print(f"Active method {self._method}")
            print(
                f"\tEntropies mean {np.mean(entropies):.3f} std {np.std(entropies):.3f}"
            )

        return infogain


class ActiveRatioTest(ActiveInfoGain):
    """Using performance ratio for acquisition.

    Args:

    """

    def __init__(self, rng_key, beta, method="mean", debug=False):
        super().__init__(rng_key, beta=beta, debug=debug)
        self._method = method

    def __call__(self, next_task, belief, all_obs, feats_keys, verbose=True):
        """Disagreement criteria.  Higher score the better.

        Score = -1 * rew(sample_w, traj_user)/rew(sample_w, traj_sample).

        Note:
            * The higher the better. i.e. in that task, observed user traj less optimal
            than sampled traj, measured by sample_w

        """
        desc = f"Computing {self._method} acquisition features"
        ## Last observation
        obs = all_obs[-1]

        if not verbose:
            desc = None
        worst_feats_sum = DictList([belief._env.max_feats_dict])
        next_feats_sum = belief.get_features_sum(next_task).prepare(feats_keys)
        user_feats_sum = obs.get_features_sum(next_task).prepare(feats_keys)
        #  shape (nweights,)
        next_costs = (
            (belief.weights * (next_feats_sum - worst_feats_sum))
            .numpy_array()
            .mean(axis=0)
        )
        user_costs = (
            (belief.weights * (user_feats_sum - worst_feats_sum))
            .numpy_array()
            .mean(axis=0)
        )
        next_rews = -1 * next_costs
        user_rews = -1 * user_costs
        ratios = self._beta * (user_rews - next_rews) / next_rews
        if self._debug:
            min_idx = np.argmin(ratios)
            print(f"Active method {self._method}")
            print(f"\tMin weight")
            for key, val in belief.weights[min_idx].items():
                print(f"\t-> {key}: {val:.3f}")
            print(f"\tRatios mean {np.mean(ratios):.3f} std {np.std(ratios):.3f}")

        if self._method == "mean":
            return -1 * np.mean(ratios)
        elif self._method == "min":
            return -1 * np.min(ratios)
        else:
            raise NotImplementedError


class ActiveRandom(ActiveInfoGain):
    """Random baseline

    Args:

    """

    def __init__(self, rng_key, method="random", debug=False):
        super().__init__(rng_key, beta=0.0, debug=debug)
        self._method = method

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def __call__(self, next_task, belief, all_obs, feats_keys, verbose=True):
        """Random score

        """
        self._rng_key, rng_random = random.split(self._rng_key)
        return random_uniform(rng_random)
