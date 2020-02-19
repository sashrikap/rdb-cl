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
import copy


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
        >>> weights.shape = (nbatch, 1)
        >>> weights.num_keys = 3
        >>> weights.append({
                'dist_cars': 1,
                'dist_obj': 2,
                'speed': 3,
            })
        >>> weights.shape = (nbatch + 1, 1)
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
        >>> features.shape = (nbatch, T, 1)
        >>> features.num_keys = 3
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
        expand_dims (bool): if True, `data` is padded with extra leading dimension (1, data_dims, ...)
            useful when later batching data
    """

    def __init__(self, data, expand_dims=False, jax=False):
        self._np = np if jax else onp
        self._jax = jax
        if isinstance(data, dict):
            if isinstance(data, DictList):
                self._jax = data._jax
            # Ensure every value is stored as array
            new_data = OrderedDict()
            if expand_dims:
                data = self._expand_dict(data)
            for key, val in data.items():
                val = self._np.array(val)
                assert len(val.shape) > 0
                new_data[key] = val
            super().__init__(new_data)
        elif (
            isinstance(data, list)
            or isinstance(data, tuple)
            or isinstance(data, onp.ndarray)
            or isinstance(data, np.ndarray)
        ):
            if expand_dims:
                data = [self._expand_dict(d) for d in data]
            data = self._stack_dict_by_keys(data)
            super().__init__(data)
        else:
            raise NotImplementedError
        self._assert_shape()

    def _stack_dict_by_keys(self, dicts):
        """Utility function."""
        if len(dicts) == 0:
            return OrderedDict()
        else:
            assert all([isinstance(d, dict) for d in dicts])
            keys = dicts[0].keys()
            out = OrderedDict()
            for key in keys:
                out[key] = self._np.stack([d[key] for d in dicts])
            return out

    def _expand_dict(self, dict_):
        """Utility function."""
        out = OrderedDict()
        for key, val in dict_.items():
            out[key] = self._np.array([val])
        return out

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
        """Return (nbatch, dim, ...)"""
        self._assert_shape()
        keys = list(self.keys())
        if len(keys) == 0:
            return ()
        else:
            first_shape = self[keys[0]].shape
            return first_shape

    @property
    def num_keys(self):
        """Return nkeys."""
        self._assert_shape()
        keys = list(self.keys())
        return len(keys)

    def append(self, dict_, axis=0):
        """Append dictionary to end."""
        assert isinstance(dict_, dict)
        for key, val in self.items():
            self[key] = self._np.concatenate([val, [dict_[key]]], axis=axis)
        self._assert_shape()

    def concat(self, dictlist, axis=0):
        """Concatenate with another DictList"""
        assert isinstance(dictlist, DictList)
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.concatenate([val, dictlist[key]], axis=axis)
        return DictList(data, jax=self._jax)

    def repeat(self, num, axis=0):
        """Repeat num times along given axis."""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.repeat(val, num, axis=axis)
        return DictList(data, jax=self._jax)

    def flatten(self):
        """Flatten values."""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = val.flatten()
        return DictList(data, jax=self._jax)

    def tile(self, num, axis=0):
        """Tile num times along existing 0-th dimension."""
        data = OrderedDict()
        assert axis < len(self.shape)
        for key, val in self.items():
            new_shape = [1] * len(val.shape)
            new_shape[axis] = num
            data[key] = self._np.tile(val, new_shape)
        return DictList(data, jax=self._jax)

    def expand_dims(self, axis=0):
        """Expand dimension."""
        data = OrderedDict()
        assert isinstance(axis, tuple) or isinstance(axis, int)
        if isinstance(axis, int):
            axis = [axis]
        for key, val in self.items():
            for ax in axis:
                val = self._np.expand_dims(val, axis=ax)
            data[key] = val
        return DictList(data, jax=self._jax)

    def squeeze(self, axis=0):
        """Suqueeze dimension."""
        data = OrderedDict()
        assert isinstance(axis, tuple) or isinstance(axis, int)
        if isinstance(axis, int):
            axis = [axis]
        for key, val in self.items():
            for ax in axis:
                val = self._np.squeeze(val, axis=ax)
            data[key] = val
        return DictList(data, jax=self._jax)

    def log(self):
        """Return log value"""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.log(val)
        return DictList(data, jax=self._jax)

    def exp(self):
        """Return log value"""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.exp(val)
        return DictList(data, jax=self._jax)

    def transpose(self):
        """Transpose every value."""
        data = OrderedDict()
        for key, val in self.items():
            data[key] = val.T
        return DictList(data, jax=self._jax)

    def sum(self, axis, keepdims=False):
        """Sum each value by axis.

        Note:
            * if val is 1-D and `keepdims=False`, output is 0-D non-DictList object

        """
        shape = self.shape
        assert axis != 0, f"Cannot sum across batch"
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.sum(val, axis=axis, keepdims=keepdims)
        if len(shape) > 1:
            # value >= 2D, resulting sum >= 1D
            return DictList(data, jax=self._jax)
        else:
            return data

    def normalize_by_key(self, key):
        """Normalize all values based on key.
        """
        data = OrderedDict()
        norm_val = self[key]
        for key, val in self.items():
            data[key] = val / norm_val
        return DictList(data, jax=self._jax)

    def normalize_across_keys(self):
        """Normalize all values such that ||w_i||_2 = 1.
        """
        this_array = self.onp_array()
        norm = onp.linalg.norm(this_array, axis=0, keepdims=True)
        this_array = this_array / norm
        new_data = self.copy()
        new_data.from_array(this_array)
        return new_data

    def copy(self):
        data = OrderedDict()
        for key, val in self.items():
            data[key] = copy.deepcopy(val)
        return DictList(data, jax=self._jax)

    def __mul__(self, data):
        """Multiply with another dictlist.
        """
        out = OrderedDict()
        if isinstance(data, DictList):
            for key, val in self.items():
                assert key in data
                out[key] = data[key] * self[key]
        else:
            for key, val in self.items():
                out[key] = data * self[key]
        return DictList(out, jax=self._jax)

    def __add__(self, dict_):
        """Add another dictlist.
        """
        assert isinstance(dict_, DictList)
        data = OrderedDict()
        for key, val in self.items():
            assert key in dict_
            data[key] = dict_[key] + self[key]
        return DictList(data, jax=self._jax)

    def __sub__(self, dict_):
        """Subtract another dictlist.
        """
        assert isinstance(dict_, DictList)
        data = OrderedDict()
        for key, val in self.items():
            assert key in dict_
            data[key] = self[key] - dict_[key]
        return DictList(data, jax=self._jax)

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
        assert axis >= 0 and axis < len(
            shape
        ), f"Cannot average axis {axis}, current DictList: nkeys={shape[0]} value {shape[1:]}"
        data = OrderedDict()
        for key, val in self.items():
            data[key] = self._np.mean(val, axis=axis, keepdims=keepdims)
        if len(shape) > 1:
            # value >= 2D, resulting mean >= 1D
            return DictList(data, jax=self._jax)
        else:
            return data

    def sort_by_keys(self, keys=None):
        """Return OrderedDict by sorting"""
        out = OrderedDict()
        if keys is None:
            keys = sorted(self.keys())
        for key in keys:
            out[key] = self[key]
        return DictList(out, jax=self._jax)

    def reshape(self, shape):
        """Reshape values"""
        out = OrderedDict()
        for key, val in self.items():
            out[key] = val.reshape(shape)
        return DictList(out, jax=self._jax)

    def prepare(self, features_keys):
        """Return copy of self, weiht keys sorted.
        """
        out = OrderedDict()
        for key in features_keys:
            if key not in self.keys():
                out[key] = self._np.zeros(self.shape)
            else:
                out[key] = self[key]
        return DictList(out, jax=self._jax)

    def numpy_array(self):
        """Return stacked values in jax.numpy

        Output:
            out (ndarray): (num_feats, n_batch)

        """
        # return np.array(list(self.values()))
        return np.array(list(self.values()))

    def clone(self, jax=False):
        """Return jax.numpy.

        Output:
            out (ndarray): (num_feats, n_batch)

        """
        # return np.array(list(self.values()))
        return DictList(self, jax=jax)

    def onp_array(self):
        """Return stacked values

        Output:
            out (ndarray): (num_feats, n_batch)

        """
        return onp.array(list(self.values()))

    def from_array(self, array):
        """Load data from array"""
        this_keys = list(self.keys())
        assert len(this_keys) == self.num_keys
        for i in range(len(this_keys)):
            self[this_keys[i]] = array[i]

    def __iter__(self):
        """Iterator to do `for d in dictlist`"""
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, key):
        """Indexing by key or by index.
        """
        if isinstance(key, str):
            # key
            val = dict.__getitem__(self, key)
        elif (
            isinstance(key, int)
            or isinstance(key, self._np.ndarray)
            or isinstance(key, np.ndarray)
            or isinstance(key, list)
            or isinstance(key, tuple)
            or isinstance(key, slice)
        ):
            # index
            output = OrderedDict()
            for k, val in self.items():
                output[k] = val[key]
            if len(self.shape) == 1:
                if isinstance(key, int):
                    # normal dict
                    return output
                else:
                    return DictList(output, jax=self._jax)
            else:
                return DictList(output, jax=self._jax)
        else:
            raise NotImplementedError
        return val
