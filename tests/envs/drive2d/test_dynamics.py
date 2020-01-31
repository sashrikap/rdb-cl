"""Test dynamics in parallel mode.

"""

import jax.numpy as np
from rdb.envs.drive2d.core.dynamics import *

xdim = 4
udim = 2
batch = 10


def run_dynamics(dynamics, x0, u0):
    batch, xdim = x0.shape
    dx_batch = dynamics(x0, u0)
    assert dx_batch.shape == (batch, xdim)
    dx_single = []
    for i, (x, u) in enumerate(zip(x0, u0)):
        assert np.allclose(dx_batch[i], dynamics(np.array([x]), np.array([u])))


def test_car_dyanmics():
    dynamics = build_car_dynamics(friction=0.2)
    x0 = np.ones((batch, xdim))
    u0 = np.ones((batch, udim))
    run_dynamics(dynamics, x0, u0)


def test_speed_dyanmics():
    dynamics = build_speed_dynamics()
    x0 = np.ones((batch, xdim))
    u0 = np.ones((batch, udim))
    dx_batch = dynamics(x0, u0)
    run_dynamics(dynamics, x0, u0)


def test_identity_dynamics():
    dynamics = build_identity_dynamics()
    x0 = np.ones((batch, xdim))
    u0 = np.ones((batch, udim))
    dx_batch = dynamics(x0, u0)
    run_dynamics(dynamics, x0, u0)


def test_fixspeed_dynamics():
    dynamics = build_fixspeed_dynamics(5)
    x0 = np.ones((batch, xdim))
    u0 = np.ones((batch, udim))
    dx_batch = dynamics(x0, u0)
    run_dynamics(dynamics, x0, u0)
