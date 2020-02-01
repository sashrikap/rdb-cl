from rdb.infer.weights import *
from collections import OrderedDict
import numpy as onp


def test_dict_init():
    data = {"a": 1, "b": 1}
    w = Weights(data)
    result = {"a": [1], "b": [1]}
    assert len(w) == 1
    for key in result.keys():
        assert onp.allclose(result[key], w[key])


def test_list_init():
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}]
    w = Weights(data)
    result = {"a": [1, 1], "b": [1, 2]}
    assert len(w) == 2
    for key in result.keys():
        assert onp.allclose(result[key], w[key])


def test_list_append():
    data = {"a": 1, "b": 1}
    w = Weights(data)
    assert len(w) == 1
    w.append({"a": 1, "b": 2})
    result = {"a": onp.array([1, 1]), "b": onp.array([1, 2])}
    assert len(w) == 2
    for key in result.keys():
        assert onp.allclose(result[key], w[key])
    w.append({"a": 2, "b": 3})
    result = {"a": onp.array([1, 1, 2]), "b": onp.array([1, 2, 3])}
    assert len(w) == 3
    for key in result.keys():
        assert onp.allclose(result[key], w[key])
    w.append({"a": 0, "b": 4})
    result = {"a": onp.array([1, 1, 2, 0]), "b": onp.array([1, 2, 3, 4])}
    assert len(w) == 4
    for key in result.keys():
        assert onp.allclose(result[key], w[key])


def test_list_index():
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 4, "b": 3}]
    w = Weights(data)
    assert len(w) == 3
    out = w[0]
    result = {"a": 1, "b": 1}
    for key in result.keys():
        assert result[key] == out[key]
    out = w[1]
    result = {"a": 1, "b": 2}
    for key in result.keys():
        assert result[key] == out[key]
    out = w[2]
    result = {"a": 4, "b": 3}
    for key in result.keys():
        assert result[key] == out[key]


def test_list_index():
    data = [{"a": 1, "b": 1}, {"a": 1, "b": 2}, {"a": 4, "b": 3}]
    w = Weights(data)
    assert len(w) == 3
    out = w[0]
    result = {"a": 1, "b": 1}


def test_list_sort():
    data = [{"b": 1, "a": 1}, {"b": 1, "a": 2}]
    w = Weights(data)
    assert len(data) == 2
    w1 = w.sort_by_keys()
    for k1, k2 in zip(list(w1.keys()), ["a", "b"]):
        assert k1 == k2

    w2 = w.sort_by_keys(["b", "a"])
    for k1, k2 in zip(list(w2.keys()), ["b", "a"]):
        assert k1 == k2
