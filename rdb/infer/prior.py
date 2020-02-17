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
import numpy as onp
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

    def __init__(self):
        raise NotImplementedError

    def add_feature(self, key):
        raise NotImplementedError

    def _build_function(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class LogNormalPrior(Prior):
    """Independent Log Uniform Prior i.e. log(val) ~ Normal.

    Args:
        log_max (float): log range (-log_max, log_max)
        feature_keys (list): initial list of features

    Out:
        out (DictList): (nkeys, num_samples)

    """

    def __init__(self, normalized_key, feature_keys, std, name=""):
        self._name = name
        self._std = std
        self._normalized_key = normalized_key
        self._initial_keys = copy.deepcopy(feature_keys)
        self._feature_keys = copy.deepcopy(feature_keys)

    def add_feature(self, key):
        if key not in self._feature_keys:
            print(f"Proposal Updated for key: {key}")
            self._feature_keys.append(key)

    @property
    def feature_keys(self):
        return self._feature_keys

    def __call__(self, num_samples, jax=True):
        assert num_samples > 0, "Must sample > 0 samples"
        output = OrderedDict()
        for key in self._feature_keys:
            if key == self._normalized_key:
                val = np.zeros((num_samples,))
            else:
                val = numpyro.sample(
                    key,
                    dist.Normal(loc=0.0, scale=self._std),
                    sample_shape=(num_samples,),
                )
            output[key] = np.exp(val)
        output = DictList(output, jax=jax)
        return output


class LogUniformPrior(Prior):
    """Independent Log Uniform Prior i.e. log(val) ~ uniform.

    Args:
        log_max (float): log range (-log_max, log_max)
        feature_keys (list): initial list of features

    Out:
        out (DictList): (nkeys, num_samples)

    """

    def __init__(self, normalized_key, feature_keys, log_max, name=""):
        self._name = name
        self._log_max = log_max
        self._normalized_key = normalized_key
        self._initial_keys = copy.deepcopy(feature_keys)
        self._feature_keys = copy.deepcopy(feature_keys)

    def add_feature(self, key):
        if key not in self._feature_keys:
            print(f"Proposal Updated for key: {key}")
            self._feature_keys.append(key)

    @property
    def feature_keys(self):
        return self._feature_keys

    def __call__(self, num_samples, jax=True):
        assert num_samples > 0, "Must sample > 0 samples"
        output = OrderedDict()
        for key in self._feature_keys:
            if key == self._normalized_key:
                val = np.zeros((num_samples,))
            else:
                max_val = self._log_max
                val = numpyro.sample(
                    key, dist.Uniform(-max_val, max_val), sample_shape=(num_samples,)
                )
            output[key] = np.exp(val)
        output = DictList(output, jax=jax)
        return output
