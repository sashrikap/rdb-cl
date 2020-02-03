"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

from jax.experimental import optimizers
from scipy.optimize import minimize, basinhopping
from rdb.optim.utils import *
from rdb.infer import *
import jax.numpy as np
import jax


# =====================================================
# ================= Optimizer classes =================
# =====================================================


class OptimizerMPC(object):
    """General MPC optimizer class.

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
        method="lbfgs",
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
        self._method = method
        self._u_shape = None
        if self._T is None:
            self._T = horizon
        if not self._replan:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        ## Rollout functions
        self.h_traj = h_traj
        self.h_csum = h_csum
        self.h_grad_u = h_grad_u

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

    def _minimize(self, fn, grad_fn, us0):
        """Optimize fn using scipy.minimize.

        Args:
            fn (fn): cost function
            grad_fn (fn): gradient function
            us0 (ndarray): initial action guess (T, nbatch, udim)

        """
        if self._method == "lbfgs":
            """L-BFGS tend to work fairly well on MPC, the down side is vectorizing it
            tends to drag down overall performance

            Parameters:
                `maxcor`: found by tuning with examples/tune/tune_mpc.

            """
            res = minimize(
                fn, us0, method="L-BFGS-B", jac=grad_fn, options={"maxcor": 20}
            )
        elif self._method == "bfgs":
            """BFGS >= L-BFGS

            Note:
                * WARNING: unbearably slow to use for vectorized high-dim

            """
            res = minimize(fn, us0, method="BFGS", jac=grad_fn)
        elif self._method == "basinhopping":
            """Global Optimization

            Basin hopping uses l-bfgs-b backbone and works at least better.

            Note:
                * WARNING: unbearably slow to compile for vectorized high-dim
                4min on horizon 10, batch 100
                * `niter` requires tuning


            """

            def fn_grad_fn(us0):
                return fn(us0), grad_fn(us0)

            kwargs = {"method": "L-BFGS-B", "jac": True}
            res = basinhopping(fn_grad_fn, us0, minimizer_kwargs=kwargs, niter=200)
        else:
            raise NotImplementedError

        info = {}
        info["cost"] = res["fun"]
        info["us"] = res["x"].reshape(us0.shape)
        if "jac" in res.keys():
            info["grad"] = res["jac"]
        else:
            info["grad"] = grad_fn(info["us"])
        return info

    def _minimize(self, fn, grad_fn, us0):
        """Optimize fn using scipy.minimize.

        Args:
            fn (fn): cost function
            grad_fn (fn): gradient function
            us0 (ndarray): initial action guess (T, nbatch, udim)

        """
        raise NotImplementedError

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
            print(f"JIT - Optimizer first compile: u0 {u_shape}")
            self._u_shape = u_shape
            t_compile = time()
        elif u_shape != self._u_shape:
            print(
                f"JIT - Optimizer recompile: u0 {u_shape}, previously {self._u_shape}"
            )
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
                res = self._minimize(csum_u_x0, grad_u_x0, us0)
                opt_u_t = res["us"]
                cmin_t = res["cost"]
                grad_u_t = res["grad"]

                if t_compile is not None:
                    print(
                        f"JIT - Optimizer finish compile in {time() - t_compile:.3f}s: u0 {self._u_shape}"
                    )
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
            res = self._minimize(csum_u_x0, grad_u_x0, us0)
            opt_u = res["us"]
            cmin = res["cost"]
            grad_u = res["grad"]

            xs = self.h_traj(x0, opt_u)
            costs = self.h_csum(x0, opt_u, weights_arr)
            u_info = {"du": grad_u, "xs": xs, "cost_fn": csum_u_x0, "costs": costs}
            if t_compile is not None:
                print(
                    f"JIT - Optimizer finish compile in {time() - t_compile:.3f}s: u0 {self._u_shape}"
                )

        return np.array(opt_u)


class OptimizerScipy(OptimizerMPC):
    """Scipy Optimizer for optimal control.

    Includes scipy-powered optimizers. Though vectorizable, the batch size affects
    individual optimization outcome quality.

    General API:
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
        method="lbfgs",
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
        super().__init__(
            h_traj,
            h_grad_u,
            h_csum,
            xdim,
            udim,
            horizon,
            replan=replan,
            T=T,
            features_keys=features_keys,
            method=method,
        )
        ## Rollout functions
        self.h_traj = self._scipy_wrapper(h_traj)
        self.h_csum = self._scipy_wrapper(h_csum)
        self.h_grad_u = self._scipy_wrapper(h_grad_u, flatten_out=True)

    def _scipy_wrapper(self, jit_fn, flatten_out=False):
        """Function to interface with scipy.optimizer, which implicitly flattens input.

        This method does two things:
            * Undo scipy.minimize's flattening
            * Return ordinary numpy oupout

        Note:
            * Inefficient way
              (horizon, nbatch, udim) -> scipy -> (horizon * nbatch * udim)
              -> jit function recompile
            * Efficient way
              (horizon, nbatch, udim) -> scipy -> (horizon * nbatch * udim)
              -> scipy_warpper ->(horizon, nbatch, udim)
              -> jit function no recompile

        This undos the effect.

        """

        def _fn(x, us, *args):
            """

            Args:
                x (ndarray): initial state
                us (ndarray): actions (horizon, nbatch, udim)

            """
            us = np.reshape(us, (self._horizon, -1, self._udim))
            out = onp.array(jit_fn(x, us, *args))
            if flatten_out:
                out = out.flatten()
            return out

        return _fn

    def _minimize(self, fn, grad_fn, us0):
        """Optimize fn using scipy.minimize.

        Args:
            fn (fn): cost function
            grad_fn (fn): gradient function
            us0 (ndarray): initial action guess (T, nbatch, udim)

        """
        if self._method == "lbfgs":
            """L-BFGS tend to work fairly well on MPC, the down side is vectorizing it
            tends to drag down overall performance

            Parameters:
                `maxcor`: found by tuning with examples/tune/tune_mpc.

            """
            res = minimize(
                fn, us0, method="L-BFGS-B", jac=grad_fn, options={"maxcor": 20}
            )
        elif self._method == "bfgs":
            """BFGS >= L-BFGS

            Note:
                * WARNING: unbearably slow to use for vectorized high-dim

            """
            res = minimize(fn, us0, method="BFGS", jac=grad_fn)
        elif self._method == "basinhopping":
            """Global Optimization

            Basin hopping uses l-bfgs-b backbone and works at least better.

            Note:
                * WARNING: unbearably slow to compile for vectorized high-dim
                4min on horizon 10, batch 100
                * `niter` requires tuning


            """

            def fn_grad_fn(us0):
                return fn(us0), grad_fn(us0)

            kwargs = {"method": "L-BFGS-B", "jac": True}
            res = basinhopping(fn_grad_fn, us0, minimizer_kwargs=kwargs, niter=200)
        else:
            raise NotImplementedError

        info = {}
        info["cost"] = res["fun"]
        info["us"] = res["x"].reshape(us0.shape)
        if "jac" in res.keys():
            info["grad"] = res["jac"]
        else:
            info["grad"] = grad_fn(info["us"])
        return info


class OptimizerJax(OptimizerMPC):
    """JAX Optimizer for optimal control.

    Includes JAX-powered vectorized optimizers.

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
        method="adam",
    ):
        super().__init__(
            h_traj,
            h_grad_u,
            h_csum,
            xdim,
            udim,
            horizon,
            replan=replan,
            T=T,
            features_keys=features_keys,
            method=method,
        )

    def _minimize(self, fn, grad_fn, us0):
        """Optimize fn using jax library.

        Args:
            fn (fn): cost function
            grad_fn (fn): gradient function
            us0 (ndarray): initial action guess (T, nbatch, udim)

        """
        if self._method == "momentum":
            # Bad, try not to use momentum
            opt_init, opt_update, get_params = optimizers.momentum(
                step_size=1e-3, mass=0.9
            )
            num_steps = 200
        elif self._method == "adam":
            opt_init, opt_update, get_params = optimizers.adam(
                step_size=3e-2, b1=0.9, b2=0.99, eps=1e-8
            )
            num_steps = 200
        else:
            raise NotImplementedError

        # Define a compiled update step
        # @jax.jit
        def step(i, opt_state):
            us = get_params(opt_state)
            g = grad_fn(us)
            return opt_update(i, g, opt_state)

        opt_state = opt_init(us0)
        for i in range(num_steps):
            opt_state = step(i, opt_state)
        us = get_params(opt_state)

        info = {}
        info["grad"] = grad_fn(us)
        info["cost"] = fn(us)
        info["us"] = us
        return info