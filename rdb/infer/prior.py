"""Utility prior functions for probabilistic inference.

Includes:
    [1] Gaussian Prior.

"""
from time import time
from rdb.infer import *
from numpyro.handlers import seed
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
import numpyro.distributions as dist
import jax.numpy as np
import numpy as np
import numpyro
import copy
import jax


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

    def __init__(self, rng_key, normalized_key, feature_keys, log_max, name=""):
        super().__init__(rng_key)
        self._name = name
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
                self._log_prior_dict[key] = dist.Uniform(
                    -self._normalize_weight, self._normalize_weight
                )
            else:
                self._log_prior_dict[key] = dist.Uniform(-self._log_max, self._log_max)
            self._log_prob_dict[key] = self._build_log_prob(key)

    def _build_log_prob(self, key):
        """Log prob function: prob = log_prob(val).

        Currently hacks around the `numpyro.dist.log_prob` slow down by assigning
        uniform distribution with np.zeros prob every where.

        TODO:
            * Deal with numpyro slow down in a better way.

        """

        def fn(val):
            # print("Inside log prob", key, val.shape)
            dist = self._log_prior_dict[key]
            # with Profiler("  log prob logval"):
            log_val = np.log(val)
            low = dist.low
            high = dist.high
            # with Profiler("  log prob logical"):
            # outside = np.logical_or(np.less(log_val, low), np.greater(log_val, high))
            outside = np.logical_or(np.less(log_val, low), np.greater(log_val, high))
            # with Profiler("  log prob out"):
            # out = np.where(outside, -np.inf, np.zeros_like(log_val))
            # out = np.where(outside, -np.inf, dist.log_prob(log_val))
            out = np.where(outside, -np.inf, np.zeros_like((log_val)))
            return out
            # return np.where(outside, -np.inf, dist.log_prob(log_val))
            # return dist.log_prob(log_val)
            # return np.zeros_like(log_val)

        return fn

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
        with jax.disable_jit():
            # with Profiler(f"Log prob {self._name} entire"):
            assert self._log_prior_dict is not None, "Must initialize with random seed"
            assert isinstance(state, DictList)
            # with Profiler(f"Log prob {self._name} add feat"):
            for key in state.keys():
                self.add_feature(key)
            # with Profiler(f"Log prob {self._name} batch"):
            nbatch = len(state)
            log_prob = np.zeros(nbatch)
            for key in self._log_prior_dict.keys():
                val = np.array(state[key])
                t1 = time()
                # if key == "dist_lanes" and pkey.shape == (603,):
                #     import pdb; pdb.set_trace()
                pkey = np.array(self._log_prob_dict[key](val))
                # pkey = np.zeros_like(val)
                # print(f"Log prob {self._name}: {key} prob {(time() - t1):.3f}")
                # t1 = time()
                # if key == self._normalized_key and np.any(np.isinf(pkey)):
                #    assert False, "Not properly normalized"
                # print(f"Log prob {key} check {(time() - t1):.3f}")
                # with Profiler(f"Log prob {self._name} {key} add"):
                # print(f"Log prob {self._name} shape", log_prob.shape, "type", type(log_prob), "pkey shape", pkey.shape, "type", type(pkey))
                log_prob += pkey
        return log_prob

    def _build_function(self):
        """Build prior function.

        Output:
            out (DictList): (nkeys, num_samples)

        """

        def prior_fn(num_samples, jax=False):
            output = {}
            for key, dist_ in self._log_prior_dict.items():
                val = numpyro.sample(key, dist_, sample_shape=(num_samples,))
                output[key] = np.exp(val)
            return DictList(output, jax=jax)

        return seed(prior_fn, self._rng_key)

    def sample(self, num_samples):
        assert num_samples > 0, "Must sample > 0 samples"
        return self(num_samples)

    def __call__(self, num_samples, jax=False):
        assert (
            self._prior_fn is not None
        ), "Need to initialize with random seed by `update_key`"

        return self._prior_fn(num_samples, jax=jax)
