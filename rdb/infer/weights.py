"""Weights class for reward design.

Thin wrapper around OrderedDict, support list-like append, concat and indexing.

"""

from rdb.optim.utils import *
from rdb.infer.utils import *
import jax.numpy as np
import numpy as onp


class Weights(dict):
    """Reward weights class.

    Underlying storage structure.
        * key -> ndarray(list)

    Args:
        data (dict): payload
        batch (bool): if True, `data` is (nbatch, data_dims, ...)
            if false, `data` is (data_dims, ...), pad leading dimension

    """

    def __init__(self, data, batch=True):
        if isinstance(data, dict):
            # Ensure every value is stored as array
            new_data = {}
            for key, val in data.items():
                if batch:
                    new_data[key] = val
                else:
                    new_data[key] = onp.array([val])
            super().__init__(new_data)
        elif isinstance(data, list):
            data = concate_dict_by_keys(data)
            super().__init__(data)
        else:
            raise NotImplementedError

    def __len__(self):
        lens = onp.array([len(val) for val in self.values()])
        if len(lens) > 0:
            assert onp.all(lens == lens[0])
            return lens[0]
        else:
            return 0

    def append(self, dict_):
        """Append dictionary to end."""
        assert isinstance(dict_, dict)
        for key, val in self.items():
            self[key] = onp.stack([val, [dict_[key]]], axis=0)

    def concat(self, weights):
        """Concatenate with another Weights"""
        assert isinstance(dict_, Weights)
        for key, val in self.items():
            self[key] = onp.stack([val, weights[key]], axis=0)

    def sort_by_keys(self, keys=None):
        """Return OrderedDict by sorting"""
        if keys is None:
            keys = sorted(self.keys())
        return Weights(sort_dict_by_keys(self, keys))

    def prepare(self, features_keys):
        """Return copy of self, weiht keys sorted.
        """
        length = len(self)
        for key in features_keys:
            if key not in self.keys():
                self[key] = onp.zeros(length)
        return self.sort_by_keys(features_keys)

    def numpy_array(self):
        """Return stacked values

        Output:
            out (ndarray): (num_feats, n_batch)

        """
        return np.array(list(self.values()))

    def __getitem__(self, key):
        if isinstance(key, str):
            val = dict.__getitem__(self, key)
        elif isinstance(key, int):
            val = select_from_dict(self, key)
        else:
            raise NotImplementedError
        return val
