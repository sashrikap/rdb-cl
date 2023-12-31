from rdb.infer.dictlist import *
from collections import OrderedDict
import pytest
import numpy as onp
import jax.numpy as np


def assert_equal(dicta, dictb):
    for key in dicta.keys():
        assert np.allclose(dicta[key], dictb[key])


@pytest.mark.parametrize("d1", [1, 4, 5])
@pytest.mark.parametrize("d2", [2, 3, 4])
def test_multi_list_init(d1, d2):
    ## Multiple list
    data = [{"a": 1, "b": 1}] * d1
    # w = DictList([data] * d2)
    # assert w.shape == (d2, d1)
    # assert w.num_keys == 2

    w = DictList([data] * d2, expand_dims=True)
    assert w.shape == (1, d2, d1)
    assert w.num_keys == 2


def test_single_list_init():
    ## Single list
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    w = DictList(data)
    assert w.shape == (2,)
    assert w.num_keys == 2


def test_dict_init():
    ## Single init
    data = {"a": 1, "b": 1}
    w = DictList(data, expand_dims=True)
    result = {"a": [1], "b": [1]}
    assert len(w) == 1
    assert w.shape == (1,)
    assert w.num_keys == 2
    assert_equal(result, w)

    data = {"a": [1], "b": [1]}
    w = DictList(data, expand_dims=True)
    result = {"a": [[1]], "b": [[1]]}
    assert w.shape == (1, 1)
    assert w.num_keys == 2
    assert_equal(result, w)
    assert_equal(result, w[:, :])

    data = {"a": [1], "b": [1]}
    w = DictList(data)
    result = {"a": [1], "b": [1]}
    assert len(w) == 1
    assert w.shape == (1,)
    assert w.num_keys == 2
    assert_equal(result, w)


def test_list_init():
    ## Single list
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    w = DictList(data)
    result = {"a": [1, 1], "b": [1, 2]}
    assert len(w) == 2
    assert w.shape == (2,)
    assert w.num_keys == 2
    assert_equal(result, w)


def test_list_append():
    data = {"a": 1, "b": 1}
    w = DictList(data, expand_dims=True)
    assert w.shape == (1,)
    assert w.num_keys == 2
    assert len(w) == 1

    w.append({"a": 1, "b": 2})
    result = {"a": onp.array([1, 1]), "b": onp.array([1, 2])}
    assert len(w) == 2
    assert w.shape == (2,)
    assert w.num_keys == 2
    assert_equal(result, w)

    w.append({"a": 2, "b": 3})
    result = {"a": onp.array([1, 1, 2]), "b": onp.array([1, 2, 3])}
    assert len(w) == 3
    assert w.shape == (3,)
    assert w.num_keys == 2
    assert_equal(result, w)

    w.append({"a": 0, "b": 4})
    result = {"a": onp.array([1, 1, 2, 0]), "b": onp.array([1, 2, 3, 4])}
    assert len(w) == 4
    assert w.shape == (4,)
    assert w.num_keys == 2
    assert_equal(result, w)

    w_concat = w.concat(w)
    assert len(w_concat) == 8
    assert w_concat.shape == (8,)
    assert w.num_keys == 2
    result_concat = {"a": onp.tile(result["a"], 2), "b": onp.tile(result["b"], 2)}
    assert_equal(result_concat, w_concat)
    assert_equal(result, w)


def test_list_index():
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 4, "b": 3}]
    w = DictList(data)
    assert len(w) == 3
    assert w.shape == (3,)
    assert w.num_keys == 2
    out = w[0]
    result = {"a": 1, "b": 1}
    assert_equal(out, result)

    out = w[1]
    result = {"a": 1, "b": 2}
    assert_equal(out, result)

    out = w[2]
    result = {"a": 4, "b": 3}
    assert_equal(out, result)

    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 4, "b": 3}]
    w = DictList(data)
    assert len(w) == 3
    assert w.shape == (3,)
    assert w.num_keys == 2
    out = w[[True, False, True]]
    result = {"a": [1, 4], "b": [1, 3]}
    assert_equal(out, result)


def test_list_index():
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 4, "b": 3}]
    w = DictList(data)
    assert len(w) == 3
    assert w.shape == (3,)
    assert w.num_keys == 2
    out = w[0]
    result = {"a": 1, "b": 1}
    assert_equal(out, result)


def test_list_sort():
    data = [{"b": 1, "a": 1}, {"b": 1, "a": 2}]
    w = DictList(data)
    assert len(data) == 2
    assert w.shape == (2,)
    assert w.num_keys == 2
    w1 = w.sort_by_keys()
    for k1, k2 in zip(list(w1.keys()), ["a", "b"]):
        assert k1 == k2

    w2 = w.sort_by_keys(["b", "a"])
    for k1, k2 in zip(list(w2.keys()), ["b", "a"]):
        assert k1 == k2


def test_prepare():
    data = [{"b": 1, "a": 1}, {"b": 1, "a": 2}]
    w = DictList(data)
    keys = ["a", "b", "c"]
    w = w.prepare(keys)
    assert w.shape == (2,)
    assert w.num_keys == 3
    result = {"a": onp.array([1, 2]), "b": onp.array([1, 1]), "c": onp.array([0, 0])}
    assert_equal(w, result)


def test_numpy_array():
    data = [{"b": 1, "a": 1}, {"b": 1, "a": 2}]
    w = DictList(data)
    assert w.shape == (2,)
    assert w.num_keys == 2
    np_array = w.numpy_array()
    assert np_array.shape == (2, 2)
    assert w.num_keys == 2
    new_data = DictList([{"b": 1, "a": 1}])
    new_data.from_array(np_array)
    assert_equal(w, new_data)


def test_onumpy_array():
    data = [{"b": 1, "a": 1}, {"b": 1, "a": 2}]
    w = DictList(data)
    assert w.shape == (2,)
    assert w.num_keys == 2
    np_array = w.numpy_array()
    assert np_array.shape == (2, 2)
    new_data = DictList([{"b": 1, "a": 1}])
    new_data.from_array(np_array)
    assert_equal(w, new_data)


def test_expand_dims():
    data = {"a": [1, 2], "b": [1, 3]}
    w = DictList(data)
    result = {"a": [1, 2], "b": [1, 3]}
    assert_equal(w, result)
    w1 = w.expand_dims(1)
    result1 = {"a": [[1], [2]], "b": [[1], [3]]}
    assert_equal(w1, result1)
    assert_equal(w1[:, :], result1)
    w2 = w.expand_dims((1, 0))
    result2 = {"a": [[[1], [2]]], "b": [[[1], [3]]]}
    assert_equal(w2, result2)
    assert_equal(w2[:, :, :], result2)


def test_expand_dims():
    data = {"a": [[1], [2]], "b": [[1], [3]]}
    w = DictList(data).squeeze(axis=1)
    result = {"a": [1, 2], "b": [1, 3]}
    assert_equal(w, result)
    assert_equal(w[:], result)
    data = {"a": [[[1], [2]]], "b": [[[1], [3]]]}
    w = DictList(data).squeeze(axis=(0, 1))
    assert_equal(w, result)
    assert_equal(w[:], result)


def test_sum():
    data = {"a": [1, 2], "b": [1, 3]}
    w = DictList(data, expand_dims=True)
    w.append({"a": [2, 1], "b": [3, 3]})
    result = {"a": [[1, 2], [2, 1]], "b": [[1, 3], [3, 3]]}
    assert w.shape == (2, 2)
    assert w.num_keys == 2
    assert_equal(result, w)

    w_sum1 = w.sum(axis=1)
    result_sum1 = {"a": [3, 3], "b": [4, 6]}
    assert_equal(w_sum1, result_sum1)

    w_sum2 = w.sum(axis=1, keepdims=True)
    result_sum2 = {"a": [[3], [3]], "b": [[4], [6]]}
    assert_equal(w_sum2, result_sum2)


def test_mean():
    data = {"a": [1, 2], "b": [1, 3]}
    w = DictList(data, expand_dims=True)
    w.append({"a": [2, 1], "b": [3, 3]})
    result = {"a": [[1, 2], [2, 1]], "b": [[1, 3], [3, 3]]}
    assert w.shape == (2, 2)
    assert w.num_keys == 2
    assert_equal(result, w)

    w_sum1 = w.sum(axis=1)
    result_sum1 = {"a": [3, 3], "b": [4, 6]}
    assert_equal(w_sum1, result_sum1)
    w_mean1 = w_sum1.mean(axis=0)
    result_mean1 = {"a": 3, "b": 5}
    assert_equal(w_mean1, result_mean1)
    w_mean1 = w_sum1.mean(axis=0, keepdims=True)
    result_mean1 = {"a": [3], "b": [5]}
    assert_equal(w_mean1, result_mean1)

    w_sum2 = w.sum(axis=1, keepdims=True)
    result_sum2 = {"a": [[3], [3]], "b": [[4], [6]]}
    assert_equal(w_sum2, result_sum2)
    w_mean2 = w_sum2.mean(axis=0)
    result_mean2 = {"a": [3], "b": [5]}
    assert_equal(w_mean2, result_mean2)

    w_mean3 = w.mean(axis=0)
    result_mean3 = {"a": [1.5, 1.5], "b": [2, 3]}
    assert_equal(w_mean3, result_mean3)
    w_mean4 = w.mean(axis=1)
    result_mean4 = {"a": [1.5, 1.5], "b": [2, 3]}


def test_transpose():
    data = {"a": [1, 2], "b": [1, 2]}
    w = DictList(data, expand_dims=True)
    w.append({"a": [1, 1], "b": [3, 3]})
    result = {"a": [[1, 1], [2, 1]], "b": [[1, 3], [2, 3]]}
    assert w.shape == (2, 2)
    assert w.num_keys == 2
    assert_equal(result, w.transpose())


def test_sum():
    dict1 = DictList({"a": [[1, 1], [2, 1]], "b": [[1, 3], [2, 3]]})
    dict2 = DictList({"a": [[1, 1], [2, 1]], "b": [[1, 3], [2, 3]]})
    result = {"a": [[2, 2], [4, 2]], "b": [[2, 6], [4, 6]]}
    assert_equal(result, dict1 + dict2)


def test_mul():
    dict1 = DictList({"a": [1, 3, 5], "b": [1, 3, 2]})
    dict2 = DictList({"a": [2, 2, 2], "b": [3, 3, 3]})
    result = {"a": [2, 6, 10], "b": [3, 9, 6]}
    assert_equal(result, dict1 * dict2)


def test_sum_values():
    dict_ = DictList({"a": [[1, 1], [2, 1]], "b": [[1, 3], [2, 3]]})
    result = [[2, 4], [4, 4]]
    assert onp.allclose(result, dict_.sum_values())


def test_repeat():
    dict_ = DictList({"a": [2, 1], "b": [1, 3]}, expand_dims=True)
    out = dict_.repeat(3, axis=0)
    result = DictList({"a": [[2, 1], [2, 1], [2, 1]], "b": [[1, 3], [1, 3], [1, 3]]})
    assert_equal(out, result)


def test_tile():
    dict_ = DictList({"a": [[2, 1]], "b": [[1, 3]]})
    out = dict_.tile(3, axis=0)
    result = DictList({"a": [[2, 1], [2, 1], [2, 1]], "b": [[1, 3], [1, 3], [1, 3]]})
    assert_equal(out, result)

    dict_ = DictList({"a": [[2, 1]], "b": [[1, 3]]})
    out = dict_.tile(3, axis=1)
    result = DictList({"a": [[2, 1, 2, 1, 2, 1]], "b": [[1, 3, 1, 3, 1, 3]]})
    assert_equal(out, result)


def test_reshape():
    dict_ = DictList({"a": [[2, 1]], "b": [[1, 3]]})
    out = dict_.reshape((2, 1))
    result = DictList({"a": [[2], [1]], "b": [[1], [3]]})
    assert_equal(out, result)


def test_iter():
    dict_ = DictList({"a": [1, 2], "b": [1, 3]})
    results = [{"a": 1, "b": 1}, {"a": 2, "b": 3}]
    n = 0
    for d in dict_:
        assert_equal(d, results[n])
        n += 1
    assert n == 2

    dict_ = DictList({"a": [[1, 2], [2, 3]], "b": [[1, 3], [1, 1]]})
    results = [{"a": [1, 2], "b": [1, 3]}, {"a": [2, 3], "b": [1, 1]}]
    n = 0
    for d in dict_:
        assert_equal(d, results[n])
        n += 1
    assert n == 2


def test_empty():
    dict_ = DictList([])
    assert len(dict_) == 0


def test_normalize_by_key():
    dict_ = DictList({"a": [1, 2], "b": [1, 3]}).normalize_by_key("a")
    results = {"a": [1, 1], "b": [1, 1.5]}
    assert_equal(dict_, results)


def test_normalize_across_keys():
    dict_ = DictList({"a": [1, 2], "b": [1, 3]}, jax=False).normalize_across_keys()
    results = {
        "a": [onp.sqrt(1 / 2.0), 2.0 / onp.sqrt(13)],
        "b": [onp.sqrt(1 / 2.0), 3.0 / onp.sqrt(13)],
    }
    assert_equal(dict_, results)

    dict_ = DictList(
        {"a": [[1, 1], [2, 2]], "b": [[1, 2], [3, 3]]}, jax=False
    ).normalize_across_keys()
    results = {
        "a": [
            [onp.sqrt(1 / 2.0), onp.sqrt(1 / 5.0)],
            [2.0 / onp.sqrt(13), 2.0 / onp.sqrt(13)],
        ],
        "b": [
            [onp.sqrt(1 / 2.0), 2.0 / onp.sqrt(5.0)],
            [3.0 / onp.sqrt(13), 3.0 / onp.sqrt(13)],
        ],
    }
    assert_equal(dict_, results)


def test_dot_product():
    dicta = DictList({"a": [1, 2], "b": [1, 3]}, jax=False)
    dictb = DictList({"a": [1, 2], "b": [1, 3]}, jax=False)
    results = np.array([2, 13])
    assert np.allclose(dicta.dot(dictb), results)
    dicta = DictList({"a": 1, "b": 2}, jax=False, expand_dims=True)
    dictb = DictList({"a": [1, 2], "b": [1, 3]}, jax=False)
    results = np.array([3, 8])
    assert np.allclose(dicta.dot(dictb), results)


def test_add_key():
    dict_ = DictList({"a": [1, 2], "b": [1, 3]}, jax=False)
    new_dict = dict_.add_key("c", [1, 2])
    results = {"a": [1, 2], "b": [1, 3], "c": [1, 2]}
    assert_equal(new_dict, results)

    new_dict = dict_.add_key("c", 1)
    results = {"a": [1, 2], "b": [1, 3], "c": [1, 1]}
    assert_equal(new_dict, results)

    new_dict = dict_.add_key("b", 1)
    results = {"a": [1, 2], "b": [1, 1]}
    assert_equal(new_dict, results)
