"""Utility proposal functions for probabilistic inference.

Includes:
    [1] Gaussian Proposal Prior.

"""
from numpyro.handlers import seed
import numpyro.distributions as dist
import jax.numpy as np
import numpyro
import copy


class Proposal(object):
    """Callable & Stateful proposal class.

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
        raise NotImplemented

    def __call__(self, state):
        """Identity prior"""
        return state


class IndGaussianProposal(Proposal):
    """Independent Gaussian Proposal.

    Args:
        proposal_var (int): gaussian std
        feature_keys (list): initial list of features

    """

    def __init__(self, rng_key, normalized_key, feature_keys, proposal_var):
        super().__init__(rng_key)
        assert (
            normalized_key in feature_keys
        ), "Feature keys must include normzlized key"
        self._proposal_var = proposal_var
        self._normalized_key = normalized_key
        self._normalize_var = 1e-6
        self._initial_keys = copy.deepcopy(feature_keys)
        self._feature_keys = copy.deepcopy(feature_keys)
        self._log_std_dict = None
        self._proposal_fn = None

    def _build_log_std(self):
        """Build independent gaussian distribution for different fields."""
        log_std = {}
        for key in self._feature_keys:
            log_std[key] = self._proposal_var
        log_std[self._normalized_key] = self._normalize_var
        return log_std

    def update_key(self, rng_key):
        """Resets feature_keys."""
        super().__init__(rng_key)
        print("Feature keys of proposal function reset.")
        # Reset feature keys, log_std and proposal function
        self._feature_keys = copy.deepcopy(self._initial_keys)
        self._log_std_dict = self._build_log_std()
        self._proposal_fn = self._build_function()

    def add_feature(self, key):
        if key not in self._feature_keys:
            print(f"Proposal Updated for key: {key}")
            self._feature_keys.append(key)
            self._log_std_dict[key] = self._proposal_var
            self._proposal_fn = self._build_function()

    def _build_function(self):
        """Build proposal function."""

        def proposal_fn(state):
            assert isinstance(state, dict), "State must be dictionary type"
            assert self._log_std_dict is not None, "Must initialize with random seed"
            keys, vals = list(state.keys()), list(state.values())
            stds = list([self._log_std_dict[k] for k in keys])
            next_vals = []
            for std, val in zip(stds, vals):
                log_val = np.log(val)
                next_log_val = numpyro.sample("next_log_val", dist.Normal(log_val, std))
                next_vals.append(np.exp(next_log_val))
            next_state = dict(zip(keys, next_vals))
            return next_state

        return seed(proposal_fn, self._rng_key)

    def __call__(self, state):
        assert (
            self._proposal_fn is not None
        ), "Need to initialize with random seed by `update_key`"
        updated = False
        for key in state:
            self.add_feature(key)

        return self._proposal_fn(state)
