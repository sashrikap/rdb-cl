"""Optimizer classes.

Wrappers around different scipy.minimize packages for MPC.

"""

import jax.experimental.optimizers as jax_optimizers
import numpyro.optim as np_optimizers
from scipy.optimize import minimize, basinhopping
from rdb.optim.utils import *
from rdb.infer import *
import jax.numpy as jnp
import time
import jax


# =====================================================
# ================= Optimizer classes =================
# =====================================================


class Optimizer(object):
    def __init__(self, xdim, udim, horizon, method):
        self._xdim = xdim
        self._udim = udim
        self._horizon = horizon
        self._method = method

    def _minimize(self):
        raise NotImplementedError


class OptimizerScipy(Optimizer):
    """Scipy Optimizer for optimal control.

    Includes scipy-powered optimizers. Though vectorizable, the batch size affects
    individual optimization outcome quality.

    General API:
        >>> actions = optimizer(x0, u0=u0, weights=weights)

    """

    def __init__(self, xdim, udim, horizon, method="lbfgs"):
        """Construct Optimizer.

        Args:
            h_grad_u (fn): horizon gradient w.r.t. u
                func(x0, us, weights) -> (T * nbatch * udim, )
                input us flatten by scipy
                output needs to be flattened
            h_csum (fn): horizon cost
                func(x0, us, weights) -> (T, nbatch, 1)
                input us flatten by scipy
            horizon (int): look-ahead horizon length
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
        super().__init__(xdim, udim, horizon, method)
        # ## Rollout functions
        # self.h_csum = self._scipy_wrapper(h_csum)
        # self.h_grad_u = self._scipy_wrapper(h_grad_u, flatten_out=True)

    def _scipy_wrapper(self, jit_fn, flatten_out=False):
        """Function to interface with scipy.optimizer, which implicitly
        flattens input.

        This method does two things:
            * Undo scipy.minimize's flattening
            * Return ordinary numpy oupout

        Note:
            -> scipy_warpper ->(horizon, nbatch, udim)
            -> jit function no recompile

        This undos the effect.

        """

        def _fn(us, *args):
            """

            Args:
                us (ndarray): actions (horizon, nbatch, udim)

            """
            us = jnp.reshape(us, (self._horizon, -1, self._udim))
            out = onp.array(jit_fn(us, *args))
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
        fn = self._scipy_wrapper(fn)
        grad_fn = self._scipy_wrapper(grad_fn, flatten_out=True)

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


class OptimizerJax(Optimizer):
    """JAX Optimizer for optimal control.

    Includes JAX-powered vectorized optimizers.

    """

    def __init__(self, xdim, udim, horizon, method="adam"):
        super().__init__(xdim, udim, horizon, method=method)

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


class OptimizerNumPyro(Optimizer):
    """NumPyro Optimizer for optimal control.

    Includes Numpyro-powered vectorized optimizers. The advantage over
    JAX optimizers is that it works better with Numpyro inference.

    """

    def __init__(self, xdim, udim, horizon, method="adam"):
        super().__init__(h_csum, xdim, udim, horizon, method=method)

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
        @jax.jit
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


optimizer_engines = {
    "scipy": OptimizerScipy,
    "jax": OptimizerJax,
    "numpyro": OptimizerNumPyro,
}
