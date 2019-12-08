import numpy as np
from rdb.infer.utils import stack_dict_values
from rdb.optim.utils import concate_dict_by_keys


def test_stack_values():
    in1 = {"a": 1, "b": 2}
    in2 = {"a": 2, "b": 4}
    out = np.array([[1, 2], [2, 4]])
    assert np.allclose(stack_dict_values([in1, in2]), out)


def test_concate_by_keys():
    in1 = {"a": [1], "b": [2]}
    in2 = {"a": [2], "b": [4]}
    out = {"a": np.array([[1], [2]]), "b": [[2], [4]]}
    out_test = concate_dict_by_keys([in1, in2])
    for key in out_test.keys():
        assert np.allclose(out[key], out_test[key])
