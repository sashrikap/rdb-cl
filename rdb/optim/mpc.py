import jax
from scipy.optimize import fmin_l_bfgs_b
from functools import partial
import jax.numpy as np
import jax.random as random
import numpy as onp

"""
Model Predictive Controllers

Includes:
[1] Shooting method optimizer
TODO:
[0] Bounded control in l-bfgs(bounds={}) or SOCP
[1] Differentiating environment
[2] CEM-Method
"""
key = random.PRNGKey(0)


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
    """Generic Optimizer.

    Example:
        >>> # General API
        >>> actions = optimizer(x0, u0=u0, weights=weights)
    """

    def __init__(
        self, h_traj_u, h_grad_u, h_cost_u, udim, horizon, replan=True, T=None
    ):
        """
        Args:
            h_traj_u (fn): func(us, x0), return horizon
            h_grad_u (fn): func(us, x0, weights), horizon gradient
            h_cost_u (fn): func(us, x0, weights), horizon cost
            replan (bool): True if replan at every step

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
        Args
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
            """
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
                # Forward 1 timestep, record 1st action
                x_t = xs_t[1]
                u0 = opt_u_t
            xs.append(x_t)
            # cmin = self.h_cost_u(opt_u, x0, weights)
            # u_info = {"du": du, "xs": xs}
            # return opt_u, cmin, u_info
            return opt_u
        else:
            """
            Shooting: Only optimize control sequence at the beginning
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


def shooting_optimizer(f_dyn, f_cost, udim, horizon, dt, replan=True, T=None):
    """Create shooting optimizer.

    Args:
        f_dyn (fn): 1 step dynamics function
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
                    `array((T, xdim)) = h_forward(xu, length)`
        >> h_costs_xu: full horizon cost function, per timestep
                     `cost = h_cost_xu(xu, length)`
        >> h_cost_u: full horizon cost function, total
                   `cost = h_cost_u(x0, u, weights)`
        >> h_grad_u: d(full horizon cost)/du,
                   `grad = h_grad_u(x0, u, weights)`

    """

    @jax.jit
    def h_forward(xu):
        """ Forward `horizon` steps """
        x, u = divide_xu(xu, udim * horizon)
        xs = [x]
        u = u.reshape(horizon, udim)
        for t in range(horizon):
            next_x = x + f_dyn(x, u[t]) * dt
            xs.append(next_x)
            x = next_x
        return np.array(xs)

    @jax.jit
    def h_costs_xu(xu, weights):
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
        xs = h_forward(xu)
        _, u = divide_xu(xu, udim * horizon)
        u = u.reshape(horizon, udim)
        for t in range(horizon):
            costs = np.append(costs, f_cost_(xs[t], u[t]))
        return costs

    h_cost_xu = jax.jit(lambda *args: np.sum(h_costs_xu(*args)))
    h_grad_xu = jax.jit(jax.grad(h_cost_xu))
    # cost_xu = lambda xu: np.sum(h_costs_xu(xu))
    # grad_xu = jax.grad(h_cost_xu)

    # Numpy-based utility for Optimizer
    def h_traj_u(x0, u):
        xu = concate_xu(x0, u)
        return to_numpy(h_forward(xu))

    def h_grad_u(x0, u, weights):
        xu = concate_xu(x0, u)
        return to_numpy(h_grad_xu(xu, weights))[-udim * horizon :]

    def h_cost_u(x0, u, weights):
        xu = concate_xu(x0, u)
        return float(h_cost_xu(xu, weights))

    return Optimizer(h_traj_u, h_grad_u, h_cost_u, udim, horizon, replan, T)
