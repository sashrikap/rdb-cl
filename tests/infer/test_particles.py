from rdb.infer.particles import Particles
from jax import random
from rdb.optim.utils import concate_dict_by_keys
import numpy as onp


def test_concate_dict_speed():
    rkey = random.PRNGKey(0)
    keys = ["a", "b", "c", "d", "e"]
    dict_ws = []
    for _ in range(500):
        dict_ws.append(dict(zip(keys, onp.random.random(5))))
    particles = Particles(rkey, None, None, None, dict_ws)
    ps = particles.resample(onp.ones(500))
    assert len(ps.weights) == 500

    concate_dict_ws = concate_dict_by_keys(dict_ws)
    particles = Particles(rkey, None, None, None, sample_concate_ws=concate_dict_ws)
    assert len(particles.weights) == 500
    ps = particles.resample(onp.ones(500))
    assert len(ps.weights) == 500
