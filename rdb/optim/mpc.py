"""Model Predictive Controllers.

Includes:
    * Shooting method optimizer (open-loop)

Optional TODO:
    * Bounded control in l-bfgs(bounds={}) or SOCP
    * Differentiating environment
    * CEM-Method

"""

from time import time
from functools import partial
from rdb.optim.utils import *
from rdb.optim.optimizers import *
from rdb.optim.runner import Runner
from rdb.exps.utils import Profiler
from jax.lax import fori_loop, scan
from jax.ops import index_update
from jax.config import config
import jax.random as random
import jax.numpy as np
import numpy as onp
import jax

key = random.PRNGKey(0)
config.update("jax_enable_x64", True)


# =====================================================
# =========== Utility functions for rollout ===========
# =====================================================


def build_forward(f_dyn, xdim, udim, dt):
    """Rollout environment, given initial x and array of u.

    Args:
        xu (ndarray): concatenate([x, us])
        f_dyn (fn): one step forward function, `x_next = f_dyn(x, u)`
        udim (int): action dimension

    """

    @jax.jit
    def roll_forward(x, us):
        # Omitting last x
        # (TODO): scipy.optimize implicitly flattens array
        us = np.reshape(us, (-1, udim))

        def step(curr_x, u):
            next_x = curr_x + f_dyn(curr_x, u) * dt
            return next_x, curr_x

        last_x, xs = scan(step, x, us)
        return xs

    return roll_forward


def build_costs(udim, roll_forward, f_cost):
    """Compute cost given start state & actions.

    Args:
        x (ndarray)
        us (ndarray)
        roll_forward (fn): full trajectory forward function, `roll_forward(x, us, length)`
        f_cost (fn): one step cost function, `f_cost(x, us, weights)`

    Note:
        * IMPORTANT: roll_forward is jitted with `length, udim, dt` arguments

    """

    @jax.jit
    def roll_costs(x, us, weights):
        vf_cost = jax.vmap(partial(f_cost, weights=weights))
        # (TODO): scipy.optimize implicitly flattens array
        us = np.reshape(us, (-1, udim))
        xs = roll_forward(x, us)
        costs = vf_cost(xs, us)
        return np.array(costs)

    return roll_costs


def build_features(udim, roll_forward, f_feat):
    """Collect trajectory features

    Args:
        x (ndarray): initial state
        us (ndarray): all actions
        roll_forward (fn): full-trajectory forward function, `x_next = f_dyn(x, u)`
        f_feat (fn): one step feature function `feats = f_feat(x, u)`

    Note:
        * IMPORTANT: roll_forward is jitted with `length, udim, dt` arguments

    Output:
        feats (dict): `feat_key[dist_car] = array(T, feat_dim)`

    """

    @jax.jit
    def roll_features(x, us):
        # (TODO): scipy.optimize implicitly flattens array
        us = np.reshape(us, (-1, udim))
        xs = roll_forward(x, us)
        feats = []
        vf_feat = jax.vmap(f_feat)
        feats = vf_feat(xs, us)
        return feats

    return roll_features


def build_mpc(env, f_cost, horizon, dt, replan=True, T=None):
    """Create MPC controller.

    Args:
        env (object): has `env.dynamics_fn`
        f_cost (fn): 1 step cost function
                    `f_cost(state, act)`, use pre-specified weight
                    `f_cost(state, act, weight)`, use weight at runtime
        horizon (int): planning horizon
        dt (float): timestep size
        replan (bool): bool, plan once or replan at every step
        T (int): trajectory length, if replan=False, must be None

    Note:
        * The following functions are moved outside of Optimizer class definition as standalone functions to speed up jax complication

    """

    ## Forward dynamics
    f_dyn = env.dynamics_fn
    f_feat = env.features_fn
    xdim, udim = env.xdim, env.udim

    if T is None:
        T = horizon

    h_forward = build_forward(f_dyn=f_dyn, xdim=xdim, udim=udim, dt=dt)
    h_costs = build_costs(roll_forward=h_forward, f_cost=f_cost, udim=udim)

    """Forward/cost/grad functions for Horizon, used in optimizer"""
    h_csum = jax.jit(lambda x0, us, weights: np.sum(h_costs(x0, us, weights)))
    h_grad = jax.jit(jax.grad(h_csum, argnums=(0, 1)))
    h_grad_u = lambda x0, us, weights: h_grad(x0, us, weights)[1]

    h_traj_u = numpy_fn(h_forward)
    h_grad_u = numpy_fn(h_grad_u)
    h_csum_u = lambda x0, us, weights: float(h_csum(x0, us, weights))
    h_costs_u = lambda x0, us, weights: h_costs(x0, us, weights)

    """Forward/cost functions for T, used in runner"""
    if T == horizon:
        # Avoid repeated jit
        t_forward = h_forward
        t_costs = h_costs
        t_costs_u = h_costs_u
    else:
        t_forward = build_forward(f_dyn=f_dyn, xdim=xdim, udim=udim, dt=dt)
        t_costs = build_costs(roll_forward=t_forward, f_cost=f_cost, udim=udim)
        t_costs_u = lambda x0, us, weights: t_costs(x0, us, weights)
    t_csum = lambda x0, us, weights: np.sum(t_costs(x0, us, weights))
    t_traj_u = numpy_fn(t_forward)
    t_csum_u = lambda x0, us, weights: float(t_csum(x0, us, weights))
    t_feats_u = build_features(roll_forward=t_forward, f_feat=f_feat, udim=udim)

    optimizer = Optimizer(
        h_traj_u=h_traj_u,
        h_grad_u=h_grad_u,
        h_csum_u=h_csum_u,
        xdim=xdim,
        udim=udim,
        horizon=horizon,
        replan=replan,
        T=T,
        features_keys=env.features_keys,
    )
    runner = Runner(
        env, roll_forward=t_traj_u, roll_costs=t_costs_u, roll_features=t_feats_u
    )

    return optimizer, runner
