"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

import jax.experimental.optimizers as jax_optimizers
import numpyro.optim as np_optimizers
from scipy.optimize import minimize, basinhopping
from rdb.optim.utils import *
from rdb.infer import *
import jax.numpy as np
import time
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
        replan=-1,
        T=None,
        features_keys=[],
        method="lbfgs",
        name="",
        test_mode=False,
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
            replan (int): replan interval, < 0 if no replan
            T (int): only in replan mode, plan for longer length than horizon

        Example:
            >>> # No initialization
            >>> actions = optimizer(x0, weights=weights)
            >>> # With initialization
            >>> actions = optimizer(x0, u0=u0, weights=weights)

        Note:
            * If `weights` is provided, it is user's reponsibility to ensure
            that cost_u & grad_u can accept `weights` as argument

        """
        self._xdim = xdim
        self._udim = udim
        self._features_keys = list(features_keys) + ["bias"]
        self._replan = replan
        self._horizon = horizon
        self._T = T
        self._name = name
        self._method = method
        self._test_mode = test_mode
        self._u_shape = None
        if self._T is None:
            self._T = horizon
        if self._replan < 0:
            assert self._T == self._horizon, "No replanning, only plan for horizon"
        else:
            assert self._horizon >= self._replan
        ## Rollout functions
        self.h_traj = h_traj
        self.h_csum = h_csum
        self.h_grad_u = h_grad_u

    @property
    def T(self):
        return self._T

    @property
    def xdim(self):
        return self._xdim

    @property
    def udim(self):
        return self._udim

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
        raise NotImplementedError

    def __call__(
        self,
        x0,
        weights,
        batch=True,
        us0=None,
        weights_arr=None,
        init="zeros",
        jax=False,
    ):
        """Run Optimizer.

        Args:
            x0 (ndarray), initial state
                shape (nbatch, xdim)
            weights (dict/DictList), weights
                shape nfeats * (nbatch,)
            us0 (ndarray), initial actions
                shape (nbatch, T, xdim)
            batch (bool), batch mode. If `true`, weights and output are batched

        Output:
            acs (ndarray): actions
                shape (nbatch, T, udim)

        """
        if weights_arr is None:
            weights_arr = (
                DictList(weights, expand_dims=not batch, jax=jax)
                .prepare(self._features_keys)
                .numpy_array()
            )
        assert len(weights_arr.shape) == 2
        assert len(x0.shape) == 2
        n_batch = len(x0)

        # Track JIT recompile
        t_compile = None
        u_shape = (n_batch, self._horizon, self._udim)
        if self._u_shape is None:
            print(f"JIT - Controller <{self._name}>")
            print(f"JIT - Controller first compile: u0 {u_shape}")
            self._u_shape = u_shape
            t_compile = time.time()
        elif u_shape != self._u_shape:
            print(f"JIT - Controller <{self._name}>")
            print(
                f"JIT - Controller recompile: u0 {u_shape}, previously {self._u_shape}"
            )
            self._u_shape = u_shape
            t_compile = time.time()

        # Initial guess
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros(u_shape)
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")

        # Pytest mode
        if self._test_mode:
            return us0

        # Reshape to T-first
        us0 = us0.swapaxes(0, 1)  # (nbatch, T, u_dim) -> (T, nbatch, u_dim)

        # Optimal Control
        opt_us, xs, grad_us = [], [], []
        x_t = x0  # initial state (nbatch, xdim)
        for t in range(self._T):
            if t == 0 or (self._replan > 0 and t % self._replan == 0):
                csum_us_xt = lambda us: self.h_csum(x_t, us, weights_arr)
                grad_us_xt = lambda us: self.h_grad_u(x_t, us, weights_arr)
                res = self._minimize(csum_us_xt, grad_us_xt, us0)
                # opt_us_t (T, nbatch, u_dim)
                opt_us_t, cmin_t, grad_us_t = res["us"], res["cost"], res["grad"]
                # xs_t (T, nbatch, x_dim)
                xs_t = self.h_traj(x_t, opt_us_t)

            if t == 0 and t_compile is not None:
                print(
                    f"JIT - Controller finish compile in {time.time() - t_compile:.3f}s: u0 {self._u_shape}"
                )
            ## Forward 1 timestep, record 1st action
            opt_us.append(opt_us_t[0])
            x_t = xs_t[0]
            xs.append(x_t)
            # grad_us.append(grad_us_t[:, 0])

            ## Pop first timestep
            opt_us_t = opt_us_t[1:]
            xs_t = xs_t[1:]
            # grad_us_t = np.delete(grad_us_t, 1, 1)

        opt_us = np.stack(opt_us, axis=1)
        # u_info = {"du": grad_us, "xs": xs, "costs": costs}
        return opt_us


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
        replan=-1,
        T=None,
        features_keys=[],
        method="lbfgs",
        name="",
        test_mode=False,
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
            horizon (int): look-ahead horizon length
            replan (int): replan interval, < 0 if no replan
            T (int): trajectory length

        Example:
            >>> # No initialization
            >>> actions = optimizer(x0, weights=weights)
            >>> # With initialization
            >>> actions = optimizer(x0, u0=u0, weights=weights)

        Note:
            * If `weights` is provided, it is user's reponsibility to
            ensure that cost_u & grad_u can accept `weights` as argument

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
            name=name,
            test_mode=test_mode,
        )
        ## Rollout functions
        self.h_traj = self._scipy_wrapper(h_traj)
        self.h_csum = self._scipy_wrapper(h_csum)
        self.h_grad_u = self._scipy_wrapper(h_grad_u, flatten_out=True)

    def _scipy_wrapper(self, jit_fn, flatten_out=False):
        """Function to interface with scipy.optimizer, which implicitly
        flattens input.

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
                `maxiter`: for MPC maxiter affects running speed a lot

            """
            res = minimize(
                fn,
                us0,
                method="L-BFGS-B",
                jac=grad_fn,
                options={"maxcor": 20, "maxiter": 120},
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
        replan=-1,
        T=None,
        features_keys=[],
        method="adam",
        name="",
        test_mode=False,
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
            name=name,
            test_mode=test_mode,
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
            opt_init, opt_update, get_params = jax_optimizers.momentum(
                step_size=1e-3, mass=0.9
            )
            num_steps = 200
        elif self._method == "adam":
            opt_init, opt_update, get_params = jax_optimizers.adam(
                step_size=3e-2, b1=0.9, b2=0.99, eps=1e-8
            )
            num_steps = 200
            # num_steps = 100
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


class OptimizerNumPyro(OptimizerMPC):
    """NumPyro Optimizer for optimal control.

    Includes Numpyro-powered vectorized optimizers. The advantage over
    JAX optimizers is that it works better with Numpyro inference.

    """

    def __init__(
        self,
        h_traj,
        h_grad_u,
        h_csum,
        xdim,
        udim,
        horizon,
        replan=-1,
        T=None,
        features_keys=[],
        method="adam",
        name="",
        test_mode=False,
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
            name=name,
            test_mode=test_mode,
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
            optim = np_optimizers.Momentum(step_size=1e-3, mass=0.9)
            num_steps = 200
        elif self._method == "adam":
            optim = np_optimizers.Adam(step_size=3e-2, b1=0.9, b2=0.99, eps=1e-8)
            num_steps = 100
        else:
            raise NotImplementedError

        get_params = lambda optim_state: optim.get_params(optim_state)
        opt_update = lambda grads, optim_state: optim.update(grads, optim_state)
        opt_init = lambda params: optim.init(params)

        # Define a compiled update step
        # @jax.jit
        def step(opt_state):
            us = get_params(opt_state)
            g = grad_fn(us)
            return opt_update(g, opt_state)

        opt_state = opt_init(us0)
        for i in range(num_steps):
            opt_state = step(opt_state)
        us = get_params(opt_state)

        info = {}
        info["grad"] = grad_fn(us)
        info["cost"] = fn(us)
        info["us"] = us
        return info
