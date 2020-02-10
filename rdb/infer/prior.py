"""Utility prior functions for probabilistic inference.

Includes:
    [1] Gaussian Prior.

"""
from rdb.infer import *
from numpyro.handlers import seed
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
import numpyro.distributions as dist
import jax.numpy as np
import numpy as onp
import numpyro
import copy


class Prior(object):
    """Callable & Stateful Prior class.

    Note:
        * Allows user to interactively update keys (e.g. add features.)

    Args:
        normalized_key (str): one key that will remain normalized.

    """

    def __init__(self, rng_key):
        self._rng_key = rng_key

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def add_feature(self, key):
        raise NotImplementedError

    def _build_function(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class LogUniformPrior(Prior):
    """Independent Log Uniform Prior i.e. log(val) ~ uniform.

    Args:
        log_max (float): log range (-log_max, log_max)
        feature_keys (list): initial list of features

    Out:
        out (DictList): (nkeys, num_samples)

    """

    def __init__(self, rng_key, normalized_key, feature_keys, log_max):
        super().__init__(rng_key)
        self._log_max = log_max
        self._normalized_key = normalized_key
        self._normalize_weight = 1e-2
        self._initial_keys = copy.deepcopy(feature_keys)
        self._feature_keys = copy.deepcopy(feature_keys)
        self._log_prior_dict = None
        self._prior_fn = None

    def _build_log_prior(self, keys=None):
        if keys is None:
            keys = self._feature_keys
        for key in keys:
            if key == self._normalized_key:
                self._log_prior_dict[key] = dist.Uniform(-self._normalize_weight, self._normalize_weight)
            else:
                self._log_prior_dict[key] = dist.Uniform(-self._log_max, self._log_max)

            def fn(val, dist=self._log_prior_dict[key]):
                log_val = np.log(val)
                low = dist.low
                high = dist.high
                outside = np.logical_or(log_val < low, log_val > high)
                return np.where(outside, -np.inf, dist.log_prob(log_val))

            self._log_prob_dict[key] = jax.jit(fn)

    def update_key(self, rng_key):
        """Resets feature_keys."""
        super().__init__(rng_key)
        print("Feature keys of prior function reset.")
        # Reset feature keys, log_std and proposal function
        self._feature_keys = copy.deepcopy(self._initial_keys)
        self._log_prior_dict = {}
        self._log_prob_dict = {}
        self._build_log_prior()
        self._prior_fn = self._build_function()

    def add_feature(self, key):
        if key not in self._feature_keys:
            print(f"Proposal Updated for key: {key}")
            self._feature_keys.append(key)
            self._build_log_prior([key])
            self._prior_fn = self._build_function()

    def log_prob(self, state):
        """Log probability of the reiceived state."""
        assert self._log_prior_dict is not None, "Must initialize with random seed"
        assert isinstance(state, DictList)
        for key in state.keys():
            self.add_feature(key)
        nbatch = len(state)
        log_prob = onp.zeros(nbatch)
        for key in self._log_prior_dict.keys():
            val = state[key]
            pkey = self._log_prob_dict[key](val)
            if key == self._normalized_key and onp.any(onp.isinf(pkey)):
                assert False, "Not properly normalized"
            log_prob += pkey
        return log_prob

    def _build_function(self):
        """Build prior function.

        Output:
            out (DictList): (nkeys, num_samples)

        """

        def prior_fn(num_samples):
            output = {}
            for key, dist_ in self._log_prior_dict.items():
                val = numpyro.sample(key, dist_, sample_shape=(num_samples,))
                output[key] = onp.exp(val)
            return DictList(output)

        return seed(prior_fn, self._rng_key)

    def sample(self, num_samples):
        assert num_samples > 0, "Must sample > 0 samples"
        return self(num_samples)

    def __call__(self, num_samples):
        assert (
            self._prior_fn is not None
        ), "Need to initialize with random seed by `update_key`"

        return self._prior_fn(num_samples)
