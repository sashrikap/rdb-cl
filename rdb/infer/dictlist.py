"""Custom thin wrapper around dict, support list-like append, concat, indexing, shape, summing, etc.

Used for representing list of dictionaries (nbatch, dict)
    * 1000 sampled weights,
    * Features of 500 rollout trajectories
    * violation of 2147 rollout features

"""

from rdb.optim.utils import *
from rdb.infer.utils import *
import jax.numpy as np
import numpy as onp


class DictList(dict):
    """Dictionary-List data structure.

    Underlying storage structure.
        * key -> ndarray(list)

    Examples:
        * Representing batch dictionary
        ```
        >>> weights = DictList({
                'dist_cars': ndarray(nbatch, 1),
                'dist_obj': ndarray(nbatch, 1),
                'speed': ndarray(nbatch, 1)
            })
        >>> len(weights) = nbatch
        >>> weights.shape = (3, nbatch, 1)
        >>> weights.append({
                'dist_cars': 1,
                'dist_obj': 2,
                'speed': 3,
            })
        >>> weights.shape = (4, nbatch, 1)
        >>> weights[3] = {
                'dist_cars': 1,
                'dist_obj': 2,
                'speed': 3,
            }
        ```

        * Representing rollout featuers
        ```
        >>> features = DictList({
                'dist_cars': ndarray(nbatch, T, 1),
                'dist_obj': ndarray(nbatch, T, 1),
                'speed': ndarray(nbatch, T, 1)
            })
        >>> len(features) = nbatch
        >>> features.shape = (3, nbatch, T, 1)
        >>> features[2] = {
                'dist_cars': ndarray(T, 1),
                'dist_obj': ndarray(T, 1),
                'speed': ndarray(T, 1)
            }
        >>> features.sum(axis=1) = DictList({
                'dist_cars': ndarray(nbatch, 1),
                'dist_obj': ndarray(nbatch, 1),
                'speed': ndarray(nbatch, 1)
            })
        ```

    Note:
        * All values must have the same shape

    Args:
        data (dict): payload
        batch (bool): if True, `data` is (nbatch, data_dims, ...)
            if false, `data` is (data_dims, ...), pad leading dimension

    """

    def __init__(self, data, expand_dims=False):
        if isinstance(data, dict):
            # Ensure every value is stored as array
            new_data = OrderedDict()
            for key, val in data.items():
                if expand_dims:
                    val = onp.array([val])
                else:
                    val = onp.array(val)
                    assert len(val.shape) > 0
                new_data[key] = val
            super().__init__(new_data)
        elif isinstance(data, list) or isinstance(data, tuple):
            data = concate_dict_by_keys(data)
            super().__init__(data)
        else:
            raise NotImplementedError
        self._assert_shape()

    def __len__(self):
        lens = onp.array([len(val) for val in self.values()])
        if len(lens) > 0:
            assert onp.all(lens == lens[0])
            return lens[0]
        else:
            return 0

    def _assert_shape(self):
        """Ensure all values have the same shape."""
        keys = list(self.keys())
        if len(keys) == 0:
            return
        else:
            first_shape = self[keys[0]].shape
            assert [self[k].shape == first_shape for k in keys]

    @property
    def shape(self):
        """Return (nkeys, nbatch, dim, ...)"""
        self._assert_shape()
        keys = list(self.keys())
        if len(keys) == 0:
            return ()
        else:
            first_shape = self[keys[0]].shape
            return tuple([len(keys)] + list(first_shape))

    def append(self, dict_):
        """Append dictionary to end."""
        assert isinstance(dict_, dict)
        for key, val in self.items():
            self[key] = onp.concatenate([val, [dict_[key]]], axis=0)
        self._assert_shape()

    def concat(self, dictlist):
        """Concatenate with another DictList"""
        assert isinstance(dictlist, DictList)
        data = OrderedDict()
        for key, val in self.items():
            data[key] = onp.concatenate([val, dictlist[key]], axis=0)
        return DictList(data)

    def repeat_expand_axis0(self, num):
        """Repeat num times along new 0-th dimension."""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = onp.repeat(val[None, :], num, axis=0)
        return DictList(data)

    def tile_axis0(self, num):
        """Tile num times along existing 0-th dimension."""
        data = OrderedDict()
        for key, val in self.items():
            new_shape = [1] * len(val.shape)
            new_shape[0] = num
            data[key] = onp.tile(val, new_shape)
        return DictList(data)

    def transpose(self):
        """Transpose every value."""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = val.T
        return DictList(data)

    def sum(self, axis, keepdims=False):
        """Sum each value by axis.

        Note:
            * if val is 1-D and `keepdims=False`, output is 0-D non-DictList object

        """
        shape = self.shape
        assert axis != 0, f"Cannot sum across batch"
        data = OrderedDict()
        for key, val in self.items():
            data[key] = onp.sum(val, axis=axis, keepdims=keepdims)
        if len(shape) > 2:
            # value >= 2D, resulting sum >= 1D
            return DictList(data)
        else:
            return data

    def __mul__(self, dict_):
        """Multiply with another dictlist.
        """
        assert isinstance(dict_, DictList)
        data = OrderedDict()
        for key, val in self.items():
            assert key in dict_
            data[key] = dict_[key] * self[key]
        return DictList(data)

    def __add__(self, dict_):
        """Add another dictlist.
        """
        assert isinstance(dict_, DictList)
        data = OrderedDict()
        for key, val in self.items():
            assert key in dict_
            data[key] = dict_[key] + self[key]
        return DictList(data)

    def __sub__(self, dict_):
        """Subtract another dictlist.
        """
        assert isinstance(dict_, DictList)
        data = OrderedDict()
        for key, val in self.items():
            assert key in dict_
            data[key] = dict_[key] - self[key]
        return DictList(data)

    def sum_values(self):
        """Sum all values.
        """
        vals = [v for v in self.values()]
        assert len(vals) > 0
        assert onp.all([v.shape == vals[0].shape for v in vals])
        return sum(vals)

    def mean(self, axis=1, keepdims=False):
        """Average each value by axis."""
        shape = self.shape
        assert axis >= 0 and axis + 1 < len(
            shape
        ), f"Cannot average axis {axis}, current DictList: nkeys={shape[0]} value {shape[1:]}"
        data = OrderedDict()
        for key, val in self.items():
            data[key] = onp.mean(val, axis=axis, keepdims=keepdims)
        if len(shape) > 2:
            # value >= 2D, resulting mean >= 1D
            return DictList(data)
        else:
            return data

    def sort_by_keys(self, keys=None):
        """Return OrderedDict by sorting"""
        if keys is None:
            keys = sorted(self.keys())
        return DictList(sort_dict_by_keys(self, keys))

    def reshape(self, shape):
        """Reshape values"""
        out = OrderedDict()
        for key, val in self.items():
            out[key] = val.reshape(shape)
        return DictList(out)

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
        """Indexing by key or by index.
        """
        if isinstance(key, str):
            # key
            val = dict.__getitem__(self, key)
        elif isinstance(key, int) or isinstance(key, onp.ndarray):
            # index
            output = {}
            idx = key
            for k, val in self.items():
                output[k] = val[idx]
            if len(self.shape) == 2:
                # normal dict
                return output
            else:
                return DictList(output)
        else:
            raise NotImplementedError
        return val
