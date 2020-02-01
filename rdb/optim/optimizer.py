"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

from scipy.optimize import minimize
from rdb.optim.utils import *
from rdb.infer import *
import jax.numpy as np
import jax


class OptimizerScipy(object):
    """Generic Optimizer for optimal control.

    Example:
        >>> # General API
        >>> actions = optimizer(x0, u0=u0, weights=weights)

    """

    def __init__(
        self,
        h_traj,
        h_grad_u,
        h_csum,
        xdim,
        udim,
        horizon,
        replan=True,
        T=None,
        features_keys=[],
    ):
        """Construct Optimizer.

        Args:
            h_traj (fn): horizion rollout
                func(x0, us) -> (T, nbatch, xdim)
            h_grad_u (fn): horizon gradient w.r.t. u
                func(x0, us, weights) -> (T * nbatch * udim, )
                input us flatten by scipy
                output needs to be flattened
            h_csum (fn): horizon cost
                func(x0, us, weights) -> (T, nbatch, 1)
                input us flatten by scipy
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
        ## Rollout functions
        self.h_traj = h_traj
        self.h_csum = h_csum
        self.h_grad_u = h_grad_u
        self._u_shape = None

        if self._T is None:
            self._T = horizon
        if not self._replan:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        return

    def cost_u(self, x0, us, weights):
        """Compute costs.

        Args:
            u (ndarray): actions (T, nbatch, udim)
            x0 (ndarray): initial state (nbatch, xdim)
            weights (ndarray): cost function weights (wdim, nbatch)

        Output:
            cost sum (ndarray): (nbatch, 1)

        """
        return self.h_csum(x0, us, weights)

    def grad_u(self, x0, us, weights):
        """Compute gradient w.r.t. u.

        Args:
            x0 (ndarray): initial state (nbatch, xdim)
            u (ndarray): actions (T, nbatch, udim)
            weights (ndarray): cost function weights (wdim, nbatch)

        Output:
            gradient (ndarray): (T, nbatch, udim)

        """
        return self.h_grad_u(x0, us, weights)

    def get_trajectory(self, x0, us):
        """Compute trajectory

        Args:
            x0 (ndarray): initial state (nbatch, xdim)
            u (ndarray): actions (T, nbatch, udim)
            weights (ndarray): cost function weights (wdim, nbatch)

        Ouput:
            xs (ndarray): (T, nbatch, xdim)

        """
        return self.h_traj(x0, us)

    def __call__(self, x0, weights, batch=True, us0=None, init="zeros"):
        """Run Optimizer.

        Args:
            x0 (ndarray), initial state (nbatch, xdim)
            weights (dict/DictList), weights
            batch (bool), batch mode. If `true`, weights and output are batched
                If `false`, weights and output are not batched

        Output:
            acs (ndarray): (T, nbatch, udim)

        """

        weights_arr = (
            DictList(weights, expand_dims=not batch)
            .prepare(self._features_keys)
            .numpy_array()
        )
        assert len(weights_arr.shape) == 2
        n_batch = weights_arr.shape[1]

        # Track JIT recompile
        t_compile = None
        u_shape = (self._horizon, n_batch, self._udim)
        if self._u_shape is None:
            print(f"Optimizer first compile: u0 {u_shape}")
            self._u_shape = u_shape
            t_compile = time()
        elif u_shape != self._u_shape:
            print(f"Optimizer recompile: u0 {u_shape}, previously {self._u_shape}")
            self._u_shape = u_shape
            t_compile = time()

        # Initial guess
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros(u_shape)
            elif init == "random":
                us0 = np.random(key, u_shape)
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")

        # Optimal Control
        if self._replan:
            ## Replan.
            ## Reoptimize control sequence at every timestep.
            opt_u, xs, du = [], [], []
            cmin = 0.0
            x_t = x0
            for t in range(self._T):
                csum_u_x0 = lambda u: self.h_csum(x_t, u, weights_arr)
                grad_u_x0 = lambda u: self.h_grad_u(x_t, u, weights_arr)
                res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
                if t_compile is not None:
                    print(
                        f"Optimizer finish compile in {time() - t_compile:.3f}s: u0 {self._u_shape}"
                    )
                opt_u_t = np.reshape(res["x"], u_shape)
                cmin_t = res["fun"]
                grad_u_t = res["jac"]
                opt_u.append(opt_u_t[0])
                xs_t = self.h_traj(x_t, opt_u_t)
                du.append(grad_u_t[0])
                ## Forward 1 timestep, record 1st action
                xs.append(x_t)
                x_t = xs_t[1]
                us0 = opt_u_t

        else:
            ## No Replan.
            ## Only optimize control sequence at the beginning
            csum_u_x0 = lambda u: self.h_csum(x0, u, weights_arr)
            grad_u_x0 = lambda u: self.h_grad_u(x0, u, weights_arr)
            res = minimize(csum_u_x0, us0, method="L-BFGS-B", jac=grad_u_x0)
            opt_u = res["x"]
            cmin = res["fun"]
            grad = res["jac"]
            opt_u = np.reshape(opt_u, u_shape)
            xs = self.h_traj(x0, opt_u)
            costs = self.h_csum(x0, opt_u, weights_arr)
            u_info = {"du": grad, "xs": xs, "cost_fn": csum_u_x0, "costs": costs}
            if t_compile is not None:
                print(
                    f"Optimizer finish compile in {time() - t_compile:.3f}s: u0 {self._u_shape}"
                )

        return np.array(opt_u)
