import jax.numpy as np
import jax
import abc

"""
Includes:
[1] Vectorizable Car dynamics (simple non-holonomic)

Conventions:
[1] State [x (horizontal), y (forward), angle, speed]
"""


def car_dynamics_fn(friction):
    """
    Usage:
    `next_x = x + delta_x(x, u)`
    Params:
    : x : state
    : u : [steer, accel]
    """

    @jax.jit
    def delta_x(x, u):
        dx = np.stack(
            [
                x[3] * np.cos(x[2]),
                x[3] * np.sin(x[2]),
                u[0] * x[3],
                u[1] - x[3] * friction,
            ]
        )
        return dx

    return delta_x


def speed_dynamics_fn():
    """
    Usage:
    `next_x = x + delta_x(x, u)`
    Params:
    : x : state
    : u : [angle, speed]
    """

    @jax.jit
    def delta_x(x, u):
        diff_u = u - x[2:]
        dx = np.concatenate([u, diff_u])
        return dx

    return delta_x


def identity_dynamics_fn():
    @jax.jit
    def delta_x(x, u):
        dx = np.zeros_like(x)
        return dx

    return delta_x


def fixspeed_dynamics_fn(fix_speed):
    @jax.jit
    def delta_x(x, u):
        diff_x = fix_speed * np.array([np.cos(x[2]), np.sin(x[2])]).T
        diff_u = np.zeros_like(diff_x)
        dx = np.concatenate([diff_x, diff_u])
        return dx

    return delta_x
