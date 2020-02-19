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
from rdb.infer.utils import random_uniform
from jax.scipy.special import logsumexp
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from rdb.optim.utils import *
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
from time import time


class ActiveInfoGain(object):
    """Information Gain.

    Args:
        model (object): IRD Model
        weight_params (dict): parameters (e.g. histogram) for calling.

    """

    def __init__(self, rng_key, model, beta, weight_params={}, debug=False):
        self._rng_key = rng_key
        self._model = model
        self._beta = beta
        self._tasks = []
        self._debug = debug
        self._method = "InfoGain"
        self._weight_params = weight_params
        self._time = None

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def __call__(self, next_task, belief, all_obs, verbose=True):
        """Information gain (negative entropy) criteria. Higher score the better.

        Note:
            * Equivalent to ranking by negative post-obs entropy -H(X|Y)
              [H(X) - H(X|Y)] - [H(X) - H(X|Y')] = H(X|Y') - H(X|Y)

        """
        desc = f"Computing {self._method} acquisition features"
        if self._time is None:
            t_start = time()
        if not verbose:
            desc = None

        #  shape nfeats * (nbatch)
        next_feats_sum = belief.get_features_sum(next_task, desc=desc).squeeze(0)

        entropies = []
        for ws_i in belief.weights:
            #  shape (nbatch,)
            new_costs = DictList(ws_i, expand_dims=True) * next_feats_sum
            new_costs = new_costs.numpy_array().mean(axis=0)

            new_rews = -1 * self._beta * new_costs
            next_probs = np.exp(new_rews - logsumexp(new_rews))
            next_belief = belief.resample(next_probs)

            # avoid gaussian entropy (causes NonSingular Matrix issue)
            # currently uses onp.ma.log which is non-differentiable
            ent = float(
                next_belief.entropy(
                    log_scale=False, method="histogram", **self._weight_params
                )
            )
            if onp.isnan(ent):
                ent = 1000
                print("WARNING: Entropy has NaN entropy value")
            entropies.append(ent)

        entropies = onp.array(entropies)
        infogain = -1 * onp.mean(entropies)

        if self._debug:
            print(f"Active method {self._method}")
            print(
                f"\tEntropies mean {onp.mean(entropies):.3f} std {onp.std(entropies):.3f}"
            )
        if self._time is None:
            self._time = time() - t_start
            print(f"Active InfoGain time: {self._time:.3f}")

        return infogain


class ActiveRatioTest(ActiveInfoGain):
    """Using performance ratio for acquisition.

    Args:

    """

    def __init__(self, rng_key, model, method="mean", debug=False):
        super().__init__(rng_key, model, beta=0.0, debug=debug)
        self._method = method

    def __call__(self, next_task, belief, all_obs, verbose=True):
        """Disagreement criteria.  Higher score the better.

        Score = -1 * rew(sample_w, traj_user)/rew(sample_w, traj_sample).

        Note:
            * The higher the better. i.e. in that task, observed user traj less optimal
            than sampled traj, measured by sample_w

        """
        desc = f"Computing {self._method} acquisition features"
        ## Last observation
        obs = all_obs[-1]

        if self._time is None:
            t_start = time()
        if not verbose:
            desc = None
        next_feats_sum = belief.get_features_sum(next_task, desc=desc)
        user_feats_sum = obs.get_features_sum(next_task)
        #  shape (nweights,)
        next_costs = (belief.weights * next_feats_sum).numpy_array().mean(axis=0)
        user_costs = (belief.weights * user_feats_sum).numpy_array().mean(axis=0)
        ratios = next_costs - user_costs

        if self._debug:
            min_idx = np.argmin(ratios)
            print(f"Active method {self._method}")
            print(f"\tMin weight")
            for key, val in belief.weights[min_idx].items():
                print(f"\t-> {key}: {val:.3f}")
            print(f"\tRatios mean {np.mean(ratios):.3f} std {np.std(ratios):.3f}")

        if self._time is None:
            self._time = time() - t_start
            print(f"Active Ratio time: {self._time:.3f}")

        if self._method == "mean":
            return np.mean(ratios)
        elif self._method == "min":
            return np.min(ratios)
        else:
            raise NotImplementedError


class ActiveRandom(ActiveInfoGain):
    """Random baseline

    Args:

    """

    def __init__(self, rng_key, model, method="random", debug=False):
        super().__init__(rng_key, model, beta=0.0, debug=debug)
        self._method = method

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def __call__(self, next_task, belief, all_obs, verbose=True):
        """Random score

        """
        self._rng_key, rng_random = random.split(self._rng_key)
        return random_uniform(rng_random)
