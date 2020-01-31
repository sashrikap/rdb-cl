"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

from scipy.optimize import minimize
from rdb.optim.utils import *
import jax.numpy as np
import jax


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
        ## Turn dict into list, in the same order as environment features_keys
        ## which later gets chained into one cost function
        weights = zero_fill_dict(weights, self._features_keys)
        weights_dict = sort_dict_by_keys(weights, self._features_keys)
        weights = np.array(list(weights_dict.values()))

        if not self._compiled:
            print("First time running optimizer, compiling with jit...")
            t_start = time()
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros((self._horizon, self._udim))
            elif init == "random":
                us0 = np.random(key, (self._horizon, self._udim))
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")
        if self._replan:
            """Replan. Reoptimize control sequence at every timestep."""
            opt_u, xs, du = [], [], []
            cmin = 0.0
            x_t = x0
            for t in range(self._T):
                csum_u_x0 = lambda u: self.h_csum_u(x_t, u, weights)
                grad_u_x0 = lambda u: self.h_grad_u(x_t, u, weights)
                res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
                if not self._compiled:
                    self._compiled = True
                    print(f"First pass compile time {time() - t_start:.3f}")
                opt_u_t = np.reshape(res["x"], (-1, self._udim))
                cmin_t = res["fun"]
                grad_u_t = res["jac"]
                opt_u.append(opt_u_t[0])
                xs_t = self.h_traj_u(x_t, opt_u_t)
                du.append(grad_u_t[0])
                ## Forward 1 timestep, record 1st action
                xs.append(x_t)
                x_t = xs_t[1]
                us0 = opt_u_t
            # Omitting last x
            return np.array(opt_u)
        else:
            """No Replan. Only optimize control sequence at the beginning."""
            # Runime cost function weights
            csum_u_x0 = lambda u: self.h_csum_u(x0, u, weights)
            grad_u_x0 = lambda u: self.h_grad_u(x0, u, weights)
            res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
            opt_u = res["x"]
            cmin = res["fun"]
            grad = res["jac"]
            opt_u = np.reshape(opt_u, (-1, self._udim))
            xs = self.h_traj_u(x0, opt_u)
            costs = self.h_csum_u(x0, opt_u, weights)
            u_info = {"du": grad, "xs": xs, "cost_fn": csum_u_x0, "costs": costs}
            if not self._compiled:
                self._compiled = True
                print(f"First pass compile time {time() - t_start:.3f}")
            return opt_u
