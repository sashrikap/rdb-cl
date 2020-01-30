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
from rdb.optim.utils import (
    multiply_dict_by_keys,
    subtract_dict_by_keys,
    concate_dict_by_keys,
)
from rdb.infer.utils import logsumexp, random_uniform
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from tqdm.auto import tqdm


class ActiveInfoGain(object):
    """Information Gain.

    Args:
        model (object): IRD Model
        params (dict): parameters (e.g. histogram) for calling.

    """

    def __init__(self, rng_key, model, beta, params={}, debug=False):
        self._rng_key = rng_key
        self._model = model
        self._beta = beta
        self._tasks = []
        self._debug = debug
        self._method = "InfoGain"
        self._params = params

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def _compute_probs(self, new_weights, next_feats_sum):
        """Computes prob of features (from original w) given new w.
        """
        new_costs = multiply_dict_by_keys(new_weights, next_feats_sum)
        log_probs = -1 * self._beta * np.sum(list(new_costs.values()), axis=0)
        denom = logsumexp(log_probs)
        probs = np.exp(log_probs - denom)
        return probs

    def __call__(self, next_task, next_task_name, belief, all_obs, verbose=True):
        """Information gain (negative entropy) criteria.

        Note:
            * Equivalent to ranking by negative post-obs entropy -H(X|Y)
              [H(X) - H(X|Y)] - [H(X) - H(X|Y')] = H(X|Y') - H(X|Y)

        """
        desc = f"Computing {self._method} acquisition features"
        if not verbose:
            desc = None
        next_feats_sum = belief.get_features_sum(next_task, next_task_name, desc=desc)

        entropies = []
        if belief.test_mode:
            entropies = np.zeros(len(belief.weights))
        else:
            for next_designer_ws in belief.weights:
                next_probs = self._compute_probs(next_designer_ws, next_feats_sum)
                next_belief = belief.resample(next_probs)
                # avoid gaussian entropy (causes NonSingular Matrix issue)
                # currently uses onp.ma.log which is non-differentiable
                ent = float(
                    next_belief.entropy(
                        bins=self._params["bins"],
                        method="histogram",
                        max_weights=self._params["max_weights"],
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

        return infogain


class ActiveRatioTest(ActiveInfoGain):
    """Using performance ratio for acquisition.

    Args:

    """

    def __init__(self, rng_key, model, method="mean", debug=False):
        super().__init__(rng_key, model, beta=0.0, debug=debug)
        self._method = method

    def _compute_log_ratios(self, weights, next_feats_sum, user_feats_sum):
        """Compares user features with belief sample features (from samplw_ws).

        Ratio = exp(next_ws @ user_feats) / exp(next_ws @ next_feats)

        """

        next_costs = multiply_dict_by_keys(weights, next_feats_sum)
        user_costs = multiply_dict_by_keys(weights, user_feats_sum)
        diff_costs = subtract_dict_by_keys(user_costs, next_costs)
        log_ratios = -1 * np.sum(list(diff_costs.values()), axis=0)
        return np.array(log_ratios)

    def __call__(self, next_task, next_task_name, belief, all_obs, verbose=True):
        """Disagreement criteria.

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
        next_feats_sum = belief.get_features_sum(next_task, next_task_name, desc=desc)
        user_feats_sum = obs.get_features_sum(next_task, next_task_name)
        weights = belief.concate_weights
        log_ratios = self._compute_log_ratios(weights, next_feats_sum, user_feats_sum)

        if self._debug:
            min_idx = np.argmin(log_ratios)
            print(f"Active method {self._method}")
            print(f"\tMin weight")
            for key, val in belief.weights[min_idx].items():
                print(f"\t-> {key}: {val:.3f}")
            print(
                f"\tRatios mean {np.mean(log_ratios):.3f} std {np.std(log_ratios):.3f}"
            )
        if self._method == "mean":
            return -1 * np.mean(log_ratios)
        elif self._method == "min":
            return -1 * np.min(log_ratios)
        else:
            raise NotImplementedError


class ActiveRandom(ActiveInfoGain):
    """Random baseline

    Args:

    """

    def __init__(self, rng_key, model, method="random", debug=False):
        super().__init__(rng_key, model, beta=0.0, debug=debug)
        self._method = method
        # self._random_uniform = None
        self._random_uniform = random_uniform

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._random_uniform = seed(random_uniform, self._rng_key)

    def __call__(self, next_task, next_task_name, belief, all_obs, verbose=True):
        """Random score

        """
        return self._random_uniform()
