import jax
from scipy.optimize import fmin_l_bfgs_b
from functools import partial
import jax.numpy as np
import numpy as onp

"""
Open-loop control methods

Includes:
[1] Shooting method optimizer
TODO:
[0] Bounded control in l-bfgs(bounds={}) or SOCP
[1] Differentiating environment
[2] CEM-Method
"""
key = jax.random.PRNGKey(0)


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


# Optimize
class Optimizer(object):
    """ Generic Optimizer """

    def __init__(self, t_traj_u, t_grad_u, t_cost_u, udim, horizon, mpc, T):
        """
        Params
        : t_traj_u : func(us, x0), return full traj
        : t_grad_u : func(us, x0, weights), full traj gradient
        : t_cost_u : func(us, x0, weights), full traj cost

        Usage
        ```optimizer(u0, x0, weights)```
        Note
            If `weights` is provided, it is user's reponsibility to
        ensure that cost_u & grad_u can accept `weights` as argument
        """
        self._udim = udim
        self._mpc = mpc
        self._horizon = horizon
        self._T = T
        self.t_traj_u = t_traj_u
        self.t_grad_u = t_grad_u
        self.t_cost_u = t_cost_u
        if self._T is None:
            self._T = horizon
        if not self._mpc:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        return

    def cost_u(self, u, x0, weights):
        """
        Params
        : u       : array of actions
        : x0      : initial state
        : weights : cost function weights
        """
        return self.t_cost_u(u, x0, weights)

    def grad_u(self, u, x0, weights):
        return self.t_grad_u(u, x0, weights)

    def get_trajectory(self, u, x0):
        return self.t_traj_u(u, x0)

    def __call__(self, u0, x0, weights=None):
        if self._mpc:
            """
            MPC: Reoptimize control sequence at every timestep
            """
            opt_u, xs, du = [], [], []
            cmin = 0.0
            x_t = x0
            for t in range(self._T):
                u0 = u0.flatten()
                xs.append(x_t)
                cost_u_x0 = partial(self.t_cost_u, x0=x_t, weights=weights)
                grad_u_x0 = partial(self.t_grad_u, x0=x_t, weights=weights)
                opt_u_t, cmin_t, info_t = fmin_l_bfgs_b(cost_u_x0, u0, grad_u_x0)
                opt_u_t = opt_u_t.reshape(-1, self._udim)
                opt_u.append(opt_u_t[0])
                xs_t = self.t_traj_u(opt_u_t, x_t)
                du.append(info_t["grad"][0])
                # Forward 1 timestep, record 1st action
                x_t = xs_t[1]
                u0 = opt_u_t
            xs.append(x_t)
            cmin = self.t_cost_u(opt_u, x0, weights)
            u_info = {"du": du, "xs": xs}
            return opt_u, cmin, u_info
        else:
            """
            Shooting: Only optimize control sequence at the beginning
            """
            u0 = u0.flatten()
            # Runime cost function weights
            cost_u_x0 = partial(self.t_cost_u, x0=x0, weights=weights)
            grad_u_x0 = partial(self.t_grad_u, x0=x0, weights=weights)
            opt_u, cmin, info = fmin_l_bfgs_b(cost_u_x0, u0, grad_u_x0)
            opt_u = opt_u.reshape(-1, self._udim)
            xu = concate_xu(x0, opt_u)
            xs = forward(xu)
            costs = self.t_costs_u(x0, opt_u)
            u_info = {
                "du": info["grad"],
                "xs": xs,
                "cost_fn": cost_u_x0,
                "costs": costs,
            }
            return opt_u, cmin, u_info


def shooting_optimizer(f_dyn, f_cost, udim, horizon, dt, mpc=False, T=None):
    """
    Create shooting optimizer

    Params
    : f_dyn   : 1 step dynamics function
    : f_cost  : 1 step cost function
                `f_cost(state, act)`, use pre-specified weight
                `f_cost(state, act, weight)`, use weight at runtime
    : udim    : action dimension
    : horizon : planning horizon
    : dt      : timestep size
    : mpc     : bool, plan once or replan at every step
    : T       : trajectory length, if mpc=False, must be None

    Note
    The following functions are moved outside of Optimizer class definition
    as standalone functions to speed up jax complication
    : t_forward  : full traj forward function
                   `array((T, xdim)) = forward(xu)`
    : t_costs_xu : full traj cost function, per timestep
                   `cost = t_cost_xu(xu)`
    : t_cost_u   : full traj cost function, total
                   `cost = t_cost_u(u, x0, weights)`
    : t_grad_u   : d(full traj cost)/du,
                   `grad = t_grad_u(u, x0, weights)`
    """

    @jax.jit
    def t_forward(xu):
        x, u = divide_xu(xu, udim * horizon)
        xs = [x]
        u = u.reshape(horizon, udim)
        for t in range(horizon):
            next_x = x + f_dyn(x, u[t]) * dt
            xs.append(next_x)
            x = next_x
        return np.array(xs)

    @jax.jit
    def t_costs_xu(xu, weights):
        """
        Params
        : xu : concatenate([x, u])
        """
        if weights is None:
            # Pre-specified weights
            f_cost_ = f_cost
        else:
            # Runime cost function weights
            f_cost_ = partial(f_cost, weights_dict=weights)
        costs = np.array([])
        xs = t_forward(xu)
        _, u = divide_xu(xu, udim * horizon)
        u = u.reshape(horizon, udim)
        for t in range(horizon):
            costs = np.append(costs, f_cost_(xs[t], u[t]))
        return costs

    t_cost_xu = jax.jit(lambda xu, weights: np.sum(t_costs_xu(xu, weights)))
    t_grad_xu = jax.jit(jax.grad(t_cost_xu))
    # cost_xu = lambda xu: np.sum(t_costs_xu(xu))
    # grad_xu = jax.grad(t_cost_xu)

    # Numpy-based utility for Optimizer
    def t_traj_u(u, x0):
        xu = concate_xu(x0, u)
        return to_numpy(t_forward(xu))

    def t_grad_u(u, x0, weights):
        xu = concate_xu(x0, u)
        return to_numpy(t_grad_xu(xu, weights))[-udim * horizon :]

    def t_cost_u(u, x0, weights):
        xu = concate_xu(x0, u)
        return float(t_cost_xu(xu, weights))

    return Optimizer(t_traj_u, t_grad_u, t_cost_u, udim, horizon, mpc, T)
