"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

from scipy.optimize import minimize
from rdb.optim.utils import *
from rdb.infer import *
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

    def __call__(self, x0, weights, batch=True, us0=None, init="zeros"):
        """Run Optimizer.

        Args:
            x0 (ndarray), initial state (nbatch, xdim)
            weights (dict/Weights), weights
            batch (bool), batch mode. If `true`, weights and output are batched
                If `false`, weights and output are not batched

        Output:
            acs (ndarray): (T, nbatch, udim)

        """

        weights_arr = (
            Weights(weights, batch=batch).prepare(self._features_keys).numpy_array()
        )
        assert len(weights_arr.shape) == 2
        n_batch = weights_arr.shape[1]

        if not self._compiled:
            print(f"First time using optimizer, input x0 shape {x0.shape}")
            t_start = time()
        u_shape = (self._horizon, n_batch, self._udim)
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros(u_shape)
            elif init == "random":
                us0 = np.random(key, u_shape)
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")
        if self._replan:
            """Replan. Reoptimize control sequence at every timestep."""
            opt_u, xs, du = [], [], []
            cmin = 0.0
            x_t = x0
            for t in range(self._T):
                csum_u_x0 = lambda u: self.h_csum_u(x_t, u, weights_arr)
                grad_u_x0 = lambda u: self.h_grad_u(x_t, u, weights_arr)
                res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
                if not self._compiled:
                    self._compiled = True
                    print(
                        f"Optimizer compile time {time() - t_start:.3f}, input x0 shape {x0.shape}"
                    )
                opt_u_t = np.reshape(res["x"], u_shape)
                cmin_t = res["fun"]
                grad_u_t = res["jac"]
                opt_u.append(opt_u_t[0])
                xs_t = self.h_traj_u(x_t, opt_u_t)
                du.append(grad_u_t[0])
                ## Forward 1 timestep, record 1st action
                xs.append(x_t)
                x_t = xs_t[1]
                us0 = opt_u_t
        else:
            """No Replan. Only optimize control sequence at the beginning."""
            # Runime cost function weights
            csum_u_x0 = lambda u: self.h_csum_u(x0, u, weights_arr)
            grad_u_x0 = lambda u: self.h_grad_u(x0, u, weights_arr)
            res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
            opt_u = res["x"]
            cmin = res["fun"]
            grad = res["jac"]
            opt_u = np.reshape(opt_u, u_shape)
            xs = self.h_traj_u(x0, opt_u)
            costs = self.h_csum_u(x0, opt_u, weights_arr)
            u_info = {"du": grad, "xs": xs, "cost_fn": csum_u_x0, "costs": costs}
            if not self._compiled:
                self._compiled = True
                print(
                    f"Optimizer compile time {time() - t_start:.3f}, input x0 shape {x0.shape}"
                )

        return np.array(opt_u)
