import numpy as np
from jax import random
from numpyro.handlers import seed
from rdb.infer.utils import stack_dict_values, random_choice
from rdb.optim.utils import concate_dict_by_keys
from rdb.exps.utils import Profiler


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


def test_random_probs():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = np.ones(3) / 3
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 4, probs, replacement=True))
    assert np.isclose(np.array(results).mean(), 2.01225)

    probs = np.array([0.6, 0.2, 0.2])
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 4, probs, replacement=True))
    assert np.isclose(np.array(results).mean(), 1.606)


def test_random_speed():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = np.random.random(500)
    arr = np.random.random(500)
    results = []
    for _ in range(10):
        with Profiler("Random choice"):
            res = random_choice_fn(np.arange(500), 500, probs, replacement=True)
            assert len(res) == 500
