"""Model Predictive Controllers.

Includes:
    * Shooting method optimizer

Optional TODO:
    * Bounded control in l-bfgs(bounds={}) or SOCP
    * Differentiating environment
    * CEM-Method

"""

import jax
from scipy.optimize import fmin_l_bfgs_b
from functools import partial
from rdb.optim.runner import Runner
from rdb.optim.utils import *
import jax.numpy as np
import jax.random as random
import numpy as onp

key = random.PRNGKey(0)

"""
Utility functions for rollout
"""


def concate_xu(x, u):
    """ f(x, u) = [x, u.flatten()] """
    xu = np.concatenate([x, onp.array(u).flatten()])
    return xu


def divide_xu(xu, udim):
    """ f([x, u.flatten()]) = x, u """
    return xu[:-udim], xu[-udim:]


def to_numpy(arr):
    """ f(x) = onp.array(x) """
    return onp.array(arr).astype(onp.float64)


"""
Rollout functions
"""


def f_forward(xu, f_dyn, udim, dt, length):
    """Forward `length` steps.

    Args:
        xu (ndarray): concatenate([x, us])
        f_dyn (fn): one step forward function, `x_next = f_dyn(x, u)`
        udim (int): action dimension
        length (int): trajectory length

    """
    x, u = divide_xu(xu, udim * length)
    xs = []
    u = u.reshape(length, udim)
    for t in range(length):
        next_x = x + f_dyn(x, u[t]) * dt
        xs.append(next_x)
        x = next_x
    xs.append(x)
    return np.array(xs)


def f_features(xu, f_forward, f_feat, udim, length):
    """Collect trajectory features

    Args:
        xu (ndarray): concatenate([x, us])
        f_forward (fn): full-trajectory forward function, `x_next = f_dyn(x, u)`
        f_feat (fn): one step feature function `feats = f_feat(x, u)`
        udim (int): action dimension
        length (int): trajectory length

    Note:
        * IMPORTANT: f_forward is jitted with `length, udim, dt` arguments

    Output:
        feats (dict): `feat_key[dist_car] = array(T, feat_dim)`

    """

    xs = f_forward(xu)
    x, u = divide_xu(xu, udim * length)
    u = u.reshape(length, udim)
    feats = []
    for t in range(length):
        feat_t = f_feat(xs[t], u[t])
        feats.append(feat_t)
    return concate_dict_by_keys(feats)


def f_costs_xu(xu, weights, f_forward, f_cost, udim, length):
    """Compute cost given start state & actions.

    Args:
        xu (ndarray): `concatenate([x, us])`
        f_forward (fn): full trajectory forward function, `f_forward(xu, length)`
        f_cost (fn): one step cost function, `f_cost(x, u, weights)`
        udim (int): action dimension
        length (int): number of timesteps

    Note:
        * IMPORTANT: f_forward is jitted with `length, udim, dt` arguments

    """
    if weights is None:
        # Pre-specified weights
        f_cost_ = f_cost
    else:
        # Runime cost function weights
        f_cost_ = partial(f_cost, weights=weights)
    costs = np.array([])
    xs = f_forward(xu)
    _, u = divide_xu(xu, udim * length)
    u = u.reshape(length, udim)
    for t in range(length):
        costs = np.append(costs, f_cost_(xs[t], u[t]))
    return costs


# Optimize
class Optimizer(object):
    """Generic Optimizer.

    Example:
        >>> # General API
        >>> actions = optimizer(x0, u0=u0, weights=weights)

    """

    def __init__(
        self, h_traj_u, h_grad_u, h_cost_u, udim, horizon, replan=True, T=None
    ):
        """Construct Optimizer.

        Args:
            h_traj_u (fn): func(us, x0), return horizon
            h_grad_u (fn): func(us, x0, weights), horizon gradient
            h_cost_u (fn): func(us, x0, weights), horizon cost
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
        self._udim = udim
        self._replan = replan
        self._horizon = horizon
        self._T = T
        self.h_traj_u = h_traj_u
        self.h_grad_u = h_grad_u
        self.h_cost_u = h_cost_u
        self._compiled = False

        if self._T is None:
            self._T = horizon
        if not self._replan:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        return

    def cost_u(self, x0, u, weights):
        """
        Args:
            u (ndarray): array of actions
            x0 (ndarray): initial state
            weights (dict): cost function weights

        """
        return self.h_cost_u(x0, u, weights)

    def grad_u(self, x0, u, weights):
        return self.h_grad_u(x0, u, weights)

    def get_trajectory(self, x0, u):
        return self.h_traj_u(x0, u)

    def __call__(self, x0, u0=None, weights=None, init="zeros"):
        if not self._compiled:
            print("First time running optimizer, compiling with jit...")
            self._compiled = True
        if u0 is None:
            if init == "zeros":
                u0 = np.zeros((self._horizon, self._udim))
            elif init == "random":
                u0 = np.random(key, (self._horizon, self._udim))
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
                u0 = u0.flatten()
                xs.append(x_t)
                cost_u_x0 = lambda u: self.h_cost_u(x_t, u, weights)
                grad_u_x0 = lambda u: self.h_grad_u(x_t, u, weights)
                opt_u_t, cmin_t, info_t = fmin_l_bfgs_b(cost_u_x0, u0, grad_u_x0)
                opt_u_t = opt_u_t.reshape(-1, self._udim)
                opt_u.append(opt_u_t[0])
                xs_t = self.h_traj_u(x_t, opt_u_t)
                du.append(info_t["grad"][0])
                ## Forward 1 timestep, record 1st action
                x_t = xs_t[1]
                u0 = opt_u_t
            xs.append(x_t)
            # cmin = self.h_cost_u(opt_u, x0, weights)
            # u_info = {"du": du, "xs": xs}
            # return opt_u, cmin, u_info
            return opt_u
        else:
            """No Replan.

            Only optimize control sequence at the beginning

            """
            u0 = u0.flatten()
            # Runime cost function weights
            cost_u_x0 = lambda u: self.h_cost_u(x0, u, weights)
            grad_u_x0 = lambda u: self.h_grad_u(x0, u, weights)
            opt_u, cmin, info = fmin_l_bfgs_b(cost_u_x0, u0, grad_u_x0)
            opt_u = opt_u.reshape(-1, self._udim)
            xs = self.h_traj_u(x0, opt_u)
            costs = self.h_cost_u(x0, opt_u, weights)
            u_info = {
                "du": info["grad"],
                "xs": xs,
                "cost_fn": cost_u_x0,
                "costs": costs,
            }
            # return opt_u, cmin, u_info
            return opt_u


def shooting_optimizer(env, f_cost, udim, horizon, dt, replan=True, T=None):
    """Create shooting optimizer.

    Args:
        env (object): has `env.dynamics_fn`
        f_cost (fn): 1 step cost function
                    `f_cost(state, act)`, use pre-specified weight
                    `f_cost(state, act, weight)`, use weight at runtime
        udim (int): action dimension
        horizon (int): planning horizon
        dt (float): timestep size
        replan (bool): bool, plan once or replan at every step
        T (int): trajectory length, if replan=False, must be None

    Note:
        * The following functions are moved outside of Optimizer class definition as standalone functions to speed up jax complication
        >> h_forward: full horizon forward function
                    `array((horizon, xdim)) = h_forward(xu)`
        >> t_forward: full T forward function
                    `array((T, xdim)) = t_forward(xu)`
        >> h_costs_xu: full horizon cost function, per timestep
                     `array(horizon,) = h_cost_xu(xu)`
        >> t_costs_xu: full T cost function, per timestep
                     `array(T,) = h_cost_xu(xu)`
        >> h_cost_u: full horizon cost function, total
                   `cost = h_cost_u(x0, us, weights)`
        >> t_cost_u: full T cost function, total
                   `cost = h_cost_u(x0, us, weights)`
        >> h_grad_u: d(full horizon cost)/du,
                   `grad = h_grad_u(x0, us, weights)`

    """

    ## Forward dynamics
    f_dyn = env.dynamics_fn
    f_feat = env.features_fn

    if T is None:
        T = horizon

    h_forward = jax.jit(
        partial(f_forward, f_dyn=f_dyn, udim=udim, dt=dt, length=horizon)
    )
    h_costs_xu = jax.jit(
        partial(
            f_costs_xu, f_forward=h_forward, f_cost=f_cost, udim=udim, length=horizon
        )
    )
    if T == horizon:
        # Avoid repeated jit
        t_forward = h_forward
        t_costs_xu = h_costs_xu
    else:
        t_forward = jax.jit(partial(f_forward, f_dyn=f_dyn, udim=udim, dt=dt, length=T))
        t_costs_xu = jax.jit(
            partial(f_costs_xu, f_forward=t_forward, f_cost=f_cost, udim=udim, length=T)
        )

    """Forward/cost/grad functions for Horizon, used in optimizer"""
    h_cost_xu = jax.jit(lambda *args: np.sum(h_costs_xu(*args)))
    h_grad_xu = jax.jit(jax.grad(h_cost_xu))
    h_traj_u = lambda x0, us: to_numpy(h_forward(concate_xu(x0, us)))
    h_grad_u = lambda x0, us, weights: to_numpy(h_grad_xu(concate_xu(x0, us), weights))[
        -udim * horizon :
    ]
    h_cost_u = lambda x0, us, weights: float(h_cost_xu(concate_xu(x0, us), weights))
    h_costs_u = lambda x0, us, weights: np.array(
        h_costs_xu(concate_xu(x0, us), weights)
    )

    """Forward/cost functions for T, used in runner"""
    t_traj_u = lambda x0, us: to_numpy(t_forward(concate_xu(x0, us)))
    t_cost_xu = jax.jit(lambda *args: np.sum(t_costs_xu(*args)))
    t_cost_u = lambda x0, us, weights: float(t_cost_xu(concate_xu(x0, us), weights))
    t_costs_u = lambda x0, us, weights: np.array(
        t_costs_xu(concate_xu(x0, us), weights)
    )
    t_feat_u = lambda x0, us: f_features(
        concate_xu(x0, us), f_forward=t_forward, f_feat=f_feat, udim=udim, length=T
    )

    optimizer = Optimizer(h_traj_u, h_grad_u, h_cost_u, udim, horizon, replan, T)
    runner = Runner(
        env, dynamics_fn=t_traj_u, cost_runtime=t_costs_u, features_fn=t_feat_u
    )

    return optimizer, runner
