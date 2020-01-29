"""Utility prior functions for probabilistic inference.

Includes:
    [1] Gaussian Prior.

"""
from numpyro.handlers import seed
import numpyro.distributions as dist
import jax.numpy as np
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

    def _build_log_prior(self):
        log_std = {}
        for key in self._feature_keys:
            log_std[key] = dist.Uniform(-self._log_max, self._log_max)
        log_std[self._normalized_key] = dist.Uniform(
            -self._normalize_weight, self._normalize_weight
        )
        return log_std

    def update_key(self, rng_key):
        """Resets feature_keys."""
        super().__init__(rng_key)
        print("Feature keys of prior function reset.")
        # Reset feature keys, log_std and proposal function
        self._feature_keys = copy.deepcopy(self._initial_keys)
        self._log_prior_dict = self._build_log_prior()
        self._prior_fn = self._build_function()

    def add_feature(self, key):
        if key not in self._feature_keys:
            updated = True
            print(f"Proposal Updated for key: {key}")
            self.add_feature(key)
            self._feature_keys.append(key)
            self._log_prior_dict[key] = dist.Uniform(-self._log_max, self._log_max)
            self._prior_fn = self._build_function()

    def _check_range(self, key, val, dist_):
        """Check the range of val against prior dist.

        Note:
            * numpyro.dist does not handle range in a "quiet" way
              e.g. `dist.Uniform(0, 1).log_prob(10)` will not give 0.
            * Let's fix this

        """
        assert isinstance(
            dist_, dist.Uniform
        ), "Only uniform distribution currently supported"

        log_val = np.log(val)
        low = dist_.low
        high = dist_.high
        # return np.where(
        #     log_val < low or log_val > high, -np.inf, dist_.log_prob(log_val)
        # )
        return dist_.log_prob(log_val)

    def log_prob(self, state):
        """Log probability of the reiceived state."""
        assert self._log_prior_dict is not None, "Must initialize with random seed"
        log_prob = 0.0
        for key, dist_ in self._log_prior_dict.items():
            val = state[key]
            log_prob += self._check_range(key, val, dist_)
        return log_prob

    def _build_function(self):
        """Build prior function."""

        def prior_fn():
            output = {}
            for key, dist_ in self._log_prior_dict.items():
                val = numpyro.sample(key, dist_)
                output[key] = np.exp(val)
            return output

        return seed(prior_fn, self._rng_key)

    def sample(self, num_samples):
        assert num_samples > 0, "Must sample > 0 samples"
        samples = []
        for _ in range(num_samples):
            samples.append(self())
        return samples

    def __call__(self):
        assert (
            self._prior_fn is not None
        ), "Need to initialize with random seed by `update_key`"

        return self._prior_fn()