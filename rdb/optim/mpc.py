"""Model Predictive Controllers.

Includes:
    * Shooting method optimizer (open-loop)

Optional TODO:
    * Bounded control in l-bfgs(bounds={}) or SOCP
    * Differentiating environment
    * CEM-Method

"""

from functools import partial
from rdb.optim.utils import *
from rdb.optim.optimizer import *
from rdb.optim.runner import Runner
from rdb.exps.utils import Profiler
from jax.lax import fori_loop, scan
from jax.ops import index_update
import jax.random as random
import jax.numpy as np
import numpy as onp
import time
import jax

key = random.PRNGKey(0)


# =====================================================
# ===================== MPC Class =====================
# =====================================================


class FiniteHorizonMPC(object):
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
        engine="jax",
        method="adam",
        name="",
        test_mode=False,
        support_batch=True,
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
            support_batch (bool): optimizer supports batch optimization

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
        self._features_keys = list(features_keys)
        self._replan = replan
        self._horizon = horizon
        self._support_batch = support_batch

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

        ## Optimizer
        optimizer_cls = optimizer_engines[engine]
        self._optimizer = optimizer_cls(xdim, udim, horizon, method)

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
        return self._optimizer._minimize(fn, grad_fn, us0)

    def _plan(self, x0, us0, weights_arr):
        """Plan for horizon.

        Args:
            x0 (ndarray): initial state
            us0 (ndarray): initial actions
            weights_arr (DictList): weights

        Note:
            * Function: self.h_csum(x0, us0[t], weights_arr)
            * Function: self.h_grad_u(x0, us0[t], weights_arr)

        """

        u_shape = us0.shape
        # Pytest mode
        if self._test_mode:
            return us0
        # # Non-batch mode
        # elif n_batch > 1 and not self._support_batch:
        #     all_acs = []
        #     for bi in range(n_batch):
        #         acs = self._plan(
        #             x0=x0[bi : bi + 1],
        #             us0=us0,
        #             weights_arr=weights_arr[:, bi],
        #         )
        #         all_acs.append(acs)
        #     return np.concatenate(all_acs, axis=0)
        else:
            # Track JIT recompile
            t_compile = None
            if self._u_shape is None:
                print(f"JIT - Controller <{self._name}>")
                print(f"JIT - Controller first compile: u0 {u_shape}")
                print(f"JIT - Controller first compile: weights {weights_arr.shape}")
                self._u_shape = u_shape
                t_compile = time.time()
            elif u_shape != self._u_shape:
                print(f"JIT - Controller <{self._name}>")
                print(
                    f"JIT - Controller recompile: u0 {u_shape}, previously {self._u_shape}"
                )
                self._u_shape = u_shape
                t_compile = time.time()

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
                shape nfeats * (nbatch,): regular
            weights_arr (ndarray)
                shape (nfeats, nbatch)
            us0 (ndarray), initial actions
                shape (nbatch, T, xdim)
            batch (bool), batch mode. If `true`, weights and output are batched

        Output:
            acs (ndarray): actions
                shape (nbatch, T, udim)

        """

        if weights_arr is None:
            ## Regular planning (nfeats, nbatch)
            ## Risk-averse planning (nfeats, nbatch, nweights)
            weights_arr = (
                DictList(weights, expand_dims=not batch, jax=jax)
                .prepare(self._features_keys)
                .numpy_array()
            )
        assert (
            len(weights_arr.shape) == 2 or len(weights_arr.shape) == 3
        ), f"Got shape {weights_arr.shape}"
        assert len(x0.shape) == 2

        # Initial guess
        n_batch = len(x0)
        u_shape = (n_batch, self._horizon, self._udim)
        if us0 is None:
            if init == "zeros":
                us0 = np.zeros(u_shape)
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")

        us_opt = self._plan(x0, us0, weights_arr)
        return us_opt


# =====================================================
# =========== Utility functions for rollout ===========
# =====================================================


def build_forward(f_dyn, xdim, udim, horizon, dt):
    """Rollout environment, given initial x and array of u.

    Args:
        f_dyn (fn): one step forward function, `x_next = f_dyn(x, u)`
        udim (int): action dimension

    Note:
        * udim, horizon, dt cannot be changed later

    """

    @jax.jit
    def roll_forward(x, us):
        """Forward trajectory.

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)

        Output:
            xs (ndarray): (horizon, nbatch, xdim)

        Note:
            Output includes the first x and omits the last x

        """

        def step(curr_x, u):
            next_x = curr_x + f_dyn(curr_x, u) * dt
            return next_x, curr_x

        last_x, xs = scan(step, x, us)
        return xs

    return roll_forward


def build_costs(udim, horizon, roll_forward, f_cost):
    """Compute cost given start state & actions.

    Args:
        roll_forward (fn): full trajectory forward function, `roll_forward(x, us, length)`
        f_cost (fn): one step cost function, `f_cost(x, us, weights)`

    Note:
        * udim, horizon, dt cannot be changed later

    """

    @jax.jit
    def roll_costs(x, us, weights):
        """Calculate trajectory costs

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            weights (ndarray): (nfeats, nbatch)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        vf_cost = jax.vmap(partial(f_cost, weights=weights))
        xs = roll_forward(x, us)
        # shape (horizon, nbatch)
        costs = vf_cost(xs, us)
        return np.array(costs)

    return roll_costs


def build_features(udim, horizon, roll_forward, f_feat, add_bias=True):
    """Collect trajectory features

    Args:
        x (ndarray): initial state
        us (ndarray): all actions
        roll_forward (fn): full-trajectory forward function
            usage: `x_next = f_dyn(x, u)`
        f_feat (fn): one step feature function
            usage: `feats = f_feat(x, u)`

    Note:
        * udim, horizon cannot be changed later

    Output:
        feats (dict)
        feat_key["dist_car"] = array(T, feat_dim)

    """

    # _no_bias_fn = lambda feats: feats
    # _add_bias_fn = lambda feats: feats.add_key("bias", 1.)
    # if add_bias:
    #     _bias_fn  = _add_bias_fn
    # else:
    #     _bias_fn = _no_bias_fn

    @jax.jit
    def roll_features(x, us):
        """Calculate trajectory features

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)

        Output:
            feats (DictList): nfeats * (horizon, nbatch)

        Note:
            * leverages `jax.vmap`'s awesome ability to map & concat dictionaries

        """

        xs = roll_forward(x, us)
        feats = []
        vf_feat = jax.vmap(f_feat)
        feats = vf_feat(xs, us)
        # return _bias_fn(feats)
        return feats

    return roll_features


# =====================================================
# =================== Build Function ==================
# =====================================================


def build_mpc(
    env,
    f_cost,
    horizon=10,
    dt=0.1,
    replan=-1,
    T=None,
    engine="scipy",
    method="lbfgs",
    name="",
    test_mode=False,
    add_bias=True,
    support_batch=True,
    build_costs=build_costs,
    mpc_cls=FiniteHorizonMPC,
    cost_args={},
):
    """Create MPC controller based on enviroment dynamics, feature
    and a cost function provided as argument.

    Usage:
        ```
        optimizer, runner = build_mpc(...)
        # weights.shape nfeats * (nbatch,)
        actions = optimizer(state, weights=weights)
        ```

    Args:
        env (object): has `env.dynamics_fn`
        f_cost (fn): 1 step cost function
            `f_cost(state, act)`, use pre-specified weight
            `f_cost(state, act, weight)`, use weight at runtime
        horizon (int): planning horizon
        dt (float): timestep size
        replan (int): replan interval, < 0 if no replan
        T (int): trajectory length, if replan=False, must be None

    Note:
        * The following functions are moved outside of Optimizer class definition as standalone functions to speed up jax complication

    """
    assert f_cost is not None, "Need to initialize environment and cost function."

    ## Forward dynamics
    f_dyn = env.dynamics_fn
    f_feat = env.features_fn
    xdim, udim = env.xdim, env.udim

    """Forward/cost/grad functions for Horizon, used in optimizer"""
    h_traj = build_forward(f_dyn, xdim, udim, horizon, dt)
    h_costs = build_costs(udim, horizon, h_traj, f_cost, **cost_args)
    h_csum = jax.jit(lambda x0, us, weights: np.sum(h_costs(x0, us, weights)))

    # Gradient w.r.t. x and u
    h_grad = jax.jit(jax.grad(h_csum, argnums=(0, 1)))
    h_grad_u = lambda x0, us, weights: h_grad(x0, us, weights)[1]

    """Forward/cost functions for T, used in runner"""
    if T is None:
        # T rollout same as horizon rollout
        T = horizon
        t_traj = h_traj
        t_costs = h_costs
    else:
        t_traj = build_forward(f_dyn, xdim, udim, horizon, dt)
        t_costs = build_costs(udim, horizon, t_traj, f_cost)

    # Rollout for t steps, optimzed every h (h <= t)
    t_csum = lambda x0, us, weights: np.sum(t_costs(x0, us, weights))
    t_feats = build_features(udim, horizon, t_traj, f_feat, add_bias=add_bias)

    # Create controller & runner
    controller = mpc_cls(
        h_traj=h_traj,
        h_grad_u=h_grad_u,
        h_csum=h_csum,
        xdim=xdim,
        udim=udim,
        horizon=horizon,
        replan=replan,
        T=T,
        features_keys=env.features_keys,
        support_batch=support_batch,
        engine=engine,
        method=method,
        name=name,
        test_mode=test_mode,
    )
    runner = Runner(
        env,
        roll_forward=t_traj,
        roll_costs=t_costs,
        roll_features=t_feats,
        name=name,
        T=T,
    )

    return controller, runner
