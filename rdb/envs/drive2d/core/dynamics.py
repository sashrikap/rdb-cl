"""Builder functions for environment dynamics.

Written in functional programming style such that they can
be jit-complied and run at light speed ;)

If anything isn't straightforward to follow, feel free to
contact hzyjerry@berkeley.edu

Includes:
    * Vectorizable Car dynamics (simple non-holonomic)

Credits:
    * Jerry Z. He 2019-2020

"""


import jax.numpy as jnp
import jax
import abc


def build_car_dynamics(friction, xdim=4, udim=2):
    """Forward Dynamics.

    Example:
        >>> next_x = x + delta_x(x, u)

    Args:
        x (ndarray): state of shape (nbatch, 4)
        u (ndarray): control of shape (nbatch, 2)

    Convention:
        * control: [steer, accel]
        * state [x (horizontal), y (forward), angle, speed]

    """

    @jax.jit
    def delta_x(x, u):
        assert len(x.shape) == 2 and x.shape[1] == xdim
        assert len(u.shape) == 2 and u.shape[1] == udim
        dx = jnp.stack(
            [
                x[:, 3] * jnp.cos(x[:, 2]),
                x[:, 3] * jnp.sin(x[:, 2]),
                u[:, 0] * x[:, 3],
                u[:, 1] - x[:, 3] * friction,
            ],
            axis=1,
        )
        return dx

    return delta_x


def build_speed_dynamics(xdim=4, udim=2):
    """Speed control dynamics.

    Example:
        >>> next_x = x + delta_x(x, u)

    Args:
        x (ndarray) : state of shape (nbatch, 4)
        u (ndarray) : control of shape (nbatch, 2)

    Convention:
        * control: [angle, speed]

    """

    @jax.jit
    def delta_x(x, u):
        assert len(x.shape) == 2 and x.shape[1] == xdim
        assert len(u.shape) == 2 and u.shape[1] == udim
        diff_u = u - x[:, 2:]
        dx = jnp.concatenate([u, diff_u], axis=1)
        return dx

    return delta_x


def build_identity_dynamics(xdim=4, udim=2):
    """Idle dynamics.

    Args:
        x (ndarray) : state of shape (nbatch, 4)
        u (ndarray) : control of shape (nbatch, 2)

    """

    @jax.jit
    def delta_x(x, u):
        assert len(x.shape) == 2 and x.shape[1] == xdim
        assert len(u.shape) == 2 and u.shape[1] == udim
        dx = jnp.zeros_like(x)
        return dx

    return delta_x


def build_fixspeed_dynamics(speed, xdim=4, udim=2):
    """Moving at fixed speed forward.

    Args:
        speed (float): forward velocity
        x (ndarray) : state of shape (nbatch, xdim)
        u (ndarray) : control of shape (nbatch, udim)

    """

    @jax.jit
    def delta_x(x, u):
        assert len(x.shape) == 2 and x.shape[1] == xdim
        assert len(u.shape) == 2 and u.shape[1] == udim

        diff_x = speed * jnp.stack([jnp.cos(x[:, 2]), jnp.sin(x[:, 2])], axis=1)
        diff_u = jnp.zeros_like(diff_x)
        dx = jnp.concatenate([diff_x, diff_u], axis=1)
        return dx

    return delta_x
