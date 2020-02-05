import numpy as onp
from time import time
from jax import random
from numpyro.handlers import seed
from rdb.infer.utils import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler


def test_stack_by_keys():
    in1 = {"a": [1], "b": [2]}
    in2 = {"a": [2], "b": [4]}
    out = {"a": onp.array([[1], [2]]), "b": [[2], [4]]}
    out_test = stack_dict_by_keys([in1, in2])
    for key in out_test.keys():
        assert onp.allclose(out[key], out_test[key])


def test_random_probs():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = onp.ones(3) / 3
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 100, probs, replacement=True))
    mean = onp.array(results).mean()
    assert mean > 1.99 and mean < 2.01

    probs = onp.array([0.6, 0.2, 0.2])
    arr = [1, 2, 3]
    results = []
    for _ in range(1000):
        results.append(random_choice_fn(arr, 4, probs, replacement=True))
    mean = onp.array(results).mean()
    assert mean > 1.59 and mean < 1.61


def test_random_speed():
    key = random.PRNGKey(0)
    random_choice_fn = seed(random_choice, key)
    probs = onp.random.random(500)
    arr = onp.random.random(500)
    results = []
    t1 = time()
    for _ in range(10):
        with Profiler("Random choice"):
            res = random_choice_fn(onp.arange(500), 500, probs, replacement=True)
            assert len(res) == 500
    print(f"Compute 10x 500 random took {time() - t1:.3f}")
