"""Model Predictive Controllers.

Includes:
    * Shooting method optimizer (open-loop)

Optional TODO:
    * Bounded control in l-bfgs(bounds={}) or SOCP
    * Differentiating environment
    * CEM-Method

"""

from functools import partial
from rdb.optim.utils import *
from rdb.optim.optimizer import *
from rdb.optim.runner import Runner
from rdb.exps.utils import Profiler
from jax.lax import fori_loop, scan
from jax.ops import index_update
import jax.random as random
import jax.numpy as np
import numpy as onp
import time
import jax

key = random.PRNGKey(0)


# =====================================================
# =========== Utility functions for rollout ===========
# =====================================================


def build_forward(f_dyn, xdim, udim, horizon, dt):
    """Rollout environment, given initial x and array of u.

    Args:
        f_dyn (fn): one step forward function, `x_next = f_dyn(x, u)`
        udim (int): action dimension

    Note:
        * udim, horizon, dt cannot be changed later

    """

    @jax.jit
    def roll_forward(x, us):
        """Forward trajectory.

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)

        Output:
            xs (ndarray): (horizon, nbatch, xdim)

        Note:
            Output includes the first x and omits the last x

        """

        def step(curr_x, u):
            next_x = curr_x + f_dyn(curr_x, u) * dt
            return next_x, curr_x

        last_x, xs = scan(step, x, us)
        return xs

    return roll_forward


def build_costs(udim, horizon, roll_forward, f_cost):
    """Compute cost given start state & actions.

    Args:
        roll_forward (fn): full trajectory forward function, `roll_forward(x, us, length)`
        f_cost (fn): one step cost function, `f_cost(x, us, weights)`

    Note:
        * udim, horizon, dt cannot be changed later

    """

    @jax.jit
    def roll_costs(x, us, weights):
        """Calculate trajectory costs

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            weights (ndarray): (nfeats, nbatch)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        vf_cost = jax.vmap(partial(f_cost, weights=weights))
        xs = roll_forward(x, us)
        costs = vf_cost(xs, us)
        return np.array(costs)

    return roll_costs


def build_features(udim, horizon, roll_forward, f_feat, add_bias=True):
    """Collect trajectory features

    Args:
        x (ndarray): initial state
        us (ndarray): all actions
        roll_forward (fn): full-trajectory forward function
            usage: `x_next = f_dyn(x, u)`
        f_feat (fn): one step feature function
            usage: `feats = f_feat(x, u)`

    Note:
        * udim, horizon cannot be changed later

    Output:
        feats (dict)
        feat_key["dist_car"] = array(T, feat_dim)

    """

    # _no_bias_fn = lambda feats: feats
    # _add_bias_fn = lambda feats: feats.add_key("bias", 1.)
    # if add_bias:
    #     _bias_fn  = _add_bias_fn
    # else:
    #     _bias_fn = _no_bias_fn

    @jax.jit
    def roll_features(x, us):
        """Calculate trajectory features

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)

        Output:
            feats (DictList): nfeats * (horizon, nbatch)

        Note:
            * leverages `jax.vmap`'s awesome ability to map & concat dictionaries

        """

        xs = roll_forward(x, us)
        feats = []
        vf_feat = jax.vmap(f_feat)
        feats = vf_feat(xs, us)
        # return _bias_fn(feats)
        return feats

    return roll_features


def build_mpc(
    env,
    f_cost,
    horizon=10,
    dt=0.1,
    replan=-1,
    T=None,
    engine="scipy",
    method="lbfgs",
    name="",
    test_mode=False,
    add_bias=True,
    build_costs=build_costs,
    cost_args={},
):
    """Create MPC controller based on enviroment dynamics, feature
    and a cost function provided as argument.

    Args:
        env (object): has `env.dynamics_fn`
        f_cost (fn): 1 step cost function
            `f_cost(state, act)`, use pre-specified weight
            `f_cost(state, act, weight)`, use weight at runtime
        horizon (int): planning horizon
        dt (float): timestep size
        replan (int): replan interval, < 0 if no replan
        T (int): trajectory length, if replan=False, must be None

    Note:
        * The following functions are moved outside of Optimizer class definition as standalone functions to speed up jax complication

    """
    assert f_cost is not None, "Need to initialize environment and cost function."

    ## Forward dynamics
    f_dyn = env.dynamics_fn
    f_feat = env.features_fn
    xdim, udim = env.xdim, env.udim

    """Forward/cost/grad functions for Horizon, used in optimizer"""
    h_traj = build_forward(f_dyn, xdim, udim, horizon, dt)
    h_costs = build_costs(udim, horizon, h_traj, f_cost, **cost_args)
    h_csum = jax.jit(lambda x0, us, weights: np.sum(h_costs(x0, us, weights)))

    # Gradient w.r.t. x and u
    h_grad = jax.jit(jax.grad(h_csum, argnums=(0, 1)))
    h_grad_u = lambda x0, us, weights: h_grad(x0, us, weights)[1]

    """Forward/cost functions for T, used in runner"""
    if T is None:
        # T rollout same as horizon rollout
        T = horizon
        t_traj = h_traj
        t_costs = h_costs
    else:
        t_traj = build_forward(f_dyn, xdim, udim, horizon, dt)
        t_costs = build_costs(udim, horizon, t_traj, f_cost)

    # Rollout for t steps, optimzed every h (h <= t)
    t_csum = lambda x0, us, weights: np.sum(t_costs(x0, us, weights))
    t_feats = build_features(udim, horizon, t_traj, f_feat, add_bias=add_bias)

    # Create optimizer & runner
    if engine == "scipy":
        optimizer_cls = OptimizerScipy
    elif engine == "jax":
        optimizer_cls = OptimizerJax
    elif engine == "numpyro":
        optimizer_cls = OptimizerNumPyro
    else:
        raise NotImplementedError

    optimizer = optimizer_cls(
        h_traj=h_traj,
        h_grad_u=h_grad_u,
        h_csum=h_csum,
        xdim=xdim,
        udim=udim,
        horizon=horizon,
        replan=replan,
        T=T,
        features_keys=env.features_keys,
        method=method,
        name=name,
        test_mode=test_mode,
    )
    runner = Runner(
        env,
        roll_forward=t_traj,
        roll_costs=t_costs,
        roll_features=t_feats,
        name=name,
        T=T,
    )

    return optimizer, runner
