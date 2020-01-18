"""Model Predictive Controllers.

Includes:
    * Shooting method optimizer (open-loop)

Optional TODO:
    * Bounded control in l-bfgs(bounds={}) or SOCP
    * Differentiating environment
    * CEM-Method

"""

from scipy.optimize import fmin_l_bfgs_b
from functools import partial
from time import time
from rdb.optim.runner import Runner
from rdb.exps.utils import Profiler
from rdb.optim.utils import *
from jax.lax import fori_loop, scan
from jax.ops import index_update
import jax
import jax.numpy as np
import jax.random as random
import numpy as onp

key = random.PRNGKey(0)

"""
Utility functions for rollout
"""


def to_numpy(arr):
    """ f(x) = onp.array(x) """
    return onp.array(arr).astype(onp.float64)


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


class Optimizer(object):
    """Generic Optimizer for optimal control.

    Example:
        >>> # General API
        >>> actions = optimizer(x0, u0=u0, weights=weights)

    """

    def __init__(
        self,
        h_traj_u,
        h_grad_u,
        h_csum_u,
        xdim,
        udim,
        horizon,
        replan=True,
        T=None,
        features_keys=[],
    ):
        """Construct Optimizer.

        Args:
            h_traj_u (fn): func(us, x0), return horizon
            h_grad_u (fn): func(us, x0, weights), horizon gradient
            h_csum_u (fn): func(us, x0, weights), horizon cost
            replan (bool): True if replan at every step
            T (int): only in replan mode, plan for longer length than horizon

        Example:
            >>> # No initialization
            >>> actions = optimizer(x0, weights=weights)
            >>> # With initialization
            >>> actions = optimizer(x0, u0=u0, weights=weights)

        Note:
            * If `weights` is provided, it is user's reponsibility to ensure that cost_u & grad_u can accept `weights` as argument

        """
        self._xdim = xdim
        self._udim = udim
        self._features_keys = features_keys
        self._replan = replan
        self._horizon = horizon
        self._T = T
        self.h_traj_u = h_traj_u
        self.h_grad_u = h_grad_u
        self.h_csum_u = h_csum_u
        self._compiled = False

        if self._T is None:
            self._T = horizon
        if not self._replan:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        return

    def cost_u(self, x0, us, weights):
        """
        Args:
            u (ndarray): array of actions
            x0 (ndarray): initial state
            weights (dict): cost function weights

        """
        return self.h_csum_u(x0, us, weights)

    def grad_u(self, x0, us, weights):
        return self.h_grad_u(x0, us, weights)

    def get_trajectory(self, x0, us):
        return self.h_traj_u(x0, us)

    def __call__(self, x0, us0=None, weights=None, init="zeros"):
        # Turn dict into sorted list
        weights_dict = sort_dict_by_keys(weights, self._features_keys)
        weights = np.array(list(weights_dict.values()))

        if not self._compiled:
            print("First time running optimizer, compiling with jit...")
            self._compiled = True
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros((self._horizon, self._udim))
            elif init == "random":
                us0 = np.random(key, (self._horizon, self._udim))
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")
        if self._replan:
            """Replan.

            Reoptimize control sequence at every timestep

            """
            opt_u, xs, du = [], [], []
            cmin = 0.0
            x_t = x0
            for t in range(self._T):
                csum_u_x0 = lambda u: self.h_csum_u(x_t, u, weights)
                grad_u_x0 = lambda u: self.h_grad_u(x_t, u, weights)
                opt_u_t, cmin_t, info_t = fmin_l_bfgs_b(csum_u_x0, us0, grad_u_x0)
                opt_u_t = np.reshape(opt_u_t, (-1, self._udim))
                opt_u.append(opt_u_t[0])
                xs_t = self.h_traj_u(x_t, opt_u_t)
                du.append(info_t["grad"][0])
                ## Forward 1 timestep, record 1st action
                xs.append(x_t)
                x_t = xs_t[1]
                us0 = opt_u_t
            # Omitting last x
            return np.array(opt_u)
        else:
            """No Replan.

            Only optimize control sequence at the beginning

            """
            # Runime cost function weights
            csum_u_x0 = lambda u: self.h_csum_u(x0, u, weights)
            grad_u_x0 = lambda u: self.h_grad_u(x0, u, weights)
            opt_u, cmin, info = fmin_l_bfgs_b(csum_u_x0, us0, grad_u_x0)
            opt_u = np.reshape(opt_u, (-1, self._udim))
            xs = self.h_traj_u(x0, opt_u)
            costs = self.h_csum_u(x0, opt_u, weights)
            u_info = {
                "du": info["grad"],
                "xs": xs,
                "cost_fn": csum_u_x0,
                "costs": costs,
            }
            return opt_u


def shooting_method(env, f_cost, horizon, dt, replan=True, T=None):
    """Create shooting optimizer.

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

    h_traj_u = lambda x0, us: to_numpy(h_forward(x0, us))
    h_grad_u = lambda x0, us, weights: to_numpy(h_grad(x0, us, weights)[1])
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
    t_traj_u = lambda x0, us: to_numpy(t_forward(x0, us))
    t_csum_u = lambda x0, us, weights: float(t_csum(x0, us, weights))
    t_feats_u = build_features(roll_forward=t_forward, f_feat=f_feat, udim=udim)

    optimizer = Optimizer(
        h_traj_u, h_grad_u, h_csum_u, xdim, udim, horizon, replan, T, env.features_keys
    )
    runner = Runner(
        env, roll_forward=t_traj_u, roll_costs=t_costs_u, roll_features=t_feats_u
    )

    return optimizer, runner
