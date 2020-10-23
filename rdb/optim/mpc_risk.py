"""Risk-Averse Model Predictive controllers.

Extends:
    * rdb.optim.mpc.py

Includes:
    * Trajectory-wise risk averse planning.
    * Step-wise risk averse planning

"""
import jax
import jax.numpy as jnp
from jax.lax import map as jmap
from jax.scipy.special import logsumexp
from functools import partial
from rdb.infer import *
from rdb.optim.mpc import FiniteHorizonMPC, build_mpc


# =====================================================
# ===================== MPC Class =====================
# =====================================================


class RiskAverseMPC(FiniteHorizonMPC):
    def __call__(
        self,
        x0,
        weights,
        us0=None,
        batch=True,
        weights_arr=None,
        init="zeros",
        jax=False,
        verbose=False,
    ):
        """Run Optimizer.

        Args:
            x0 (ndarray), initial state
                shape (nbatch, xdim)
            weights (dict/DictList), weights
                shape nfeats * (nrisk): non-batch, risk averse
                shape nfeats * (nbatch, nrisk): batch risk averse
            weights_arr (ndarray)
                shape (nfeats, nbatch, nrisk): risk averse
            us0 (ndarray), initial actions
                shape (nbatch, T, xdim)
            output are batched

        Output:
            acs (ndarray): actions
                shape (nbatch, T, udim)

        """
        if weights_arr is None:
            ## Risk-averse planning (nfeats, nbatch, nweights)
            weights_arr = (
                DictList(weights, expand_dims=not batch, jax=jax)
                .prepare(self._features_keys)
                .numpy_array()
            )
        assert len(weights_arr.shape) == 3, f"Got shape {weights_arr.shape}"

        # Initial guess
        n_batch = len(x0)
        u_shape = (n_batch, self._horizon, self._udim)
        if us0 is None:
            if init == "zeros":
                us0 = jnp.ones(u_shape) * 1e-5
            else:
                raise NotImplementedError(f"Initialization undefined for '{init}'")
        us_opt = self._plan(x0, us0, weights_arr, verbose=verbose)
        return us_opt


# =====================================================
# =========== Utility functions for rollout ===========
# =====================================================


def build_multi_costs(
    udim, horizon, roll_forward, f_cost, mode="stepwise", softmax=True
):
    """Compute cost given start state, actions and a list of cost coefficients.

    Args:

        mode (str): "stepwise" or "trajwise"
        softmax (bool): compute soft max over a set of weights

    """

    assert mode in {"stepwise", "trajwise"}

    def _soft_max_ratio(costs, axis=None):
        costs = jnp.array(costs)
        return jnp.exp(costs - logsumexp(costs, axis=axis))

    def _hard_max_ratio(costs, axis=None):
        costs = jnp.array(costs)
        idx = jnp.argmax(costs, axis=axis)
        ratio = jnp.zeros_like(costs)
        ratio[idx] = 1.0
        return ratio

    if softmax:
        _max_ratio = _soft_max_ratio
    else:
        _max_ratio = _hard_max_ratio

    def roll_cost(weights, x, us):
        """Calculate trajectory costs

        Args:
            weights (ndarray): (nfeats, nbatch)
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        vf_cost = jax.vmap(partial(f_cost, weights=weights))
        xs = roll_forward(x, us)
        # shape (horizon, nbatch)
        costs = vf_cost(xs, us)
        return jnp.array(costs)

    @jax.jit
    def roll_stepwise_costs(x, us, all_weights):
        """Calculate step-wise trajectory costs

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            all_weights (ndarray): (nfeats, nbatch, nweights)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        assert len(all_weights.shape) == 3, f"Got shape {all_weights.shape}"
        proll_cost = partial(roll_cost, x=x, us=us)
        #  shape (nweights, nfeats, nbatch)
        all_weights = jnp.rollaxis(all_weights, 2, 0)
        # all_weights = jnp.swapaxes(all_weights, 0, 2)
        # all_weights = jnp.swapaxes(all_weights, 1, 2)
        #  shape (nweights, horizon, nbatch)
        costs = jnp.array(jmap(proll_cost, all_weights))
        assert len(costs.shape) == 3
        ratio = _max_ratio(costs, axis=0)
        return jnp.sum(ratio * costs, axis=0)

    @jax.jit
    def roll_trajwise_costs(x, us, all_weights):
        """Calculate trajectory features

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            all_weights (ndarray): (nfeats, nbatch, nweights)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        assert len(all_weights.shape) == 3, f"Got shape {all_weights.shape}"
        proll_cost = partial(roll_cost, x=x, us=us)
        #  shape: (nweights, horizon, nbatch)
        # all_weights = jnp.rollaxis(all_weights, 2, 0)
        all_weights = jnp.swapaxes(all_weights, 0, 2)
        all_weights = jnp.swapaxes(all_weights, 1, 2)
        #  shape (nweights, horizon, nbatch)
        costs = jnp.array(jmap(proll_cost, all_weights))
        assert len(costs.shape) == 3
        csums = jnp.sum(costs, axis=(1, 2))
        ratio = _max_ratio(csums, axis=0)
        #   shape (nweights, 1, 1)
        ratio = jnp.expand_dims(jnp.expand_dims(ratio, 1), 2)
        return jnp.sum(ratio * costs, axis=0)

    if mode == "stepwise":
        return roll_stepwise_costs
    elif mode == "trajwise":
        return roll_trajwise_costs
    else:
        raise NotImplementedError


# =====================================================
# =================== Build Function ==================
# =====================================================


def build_risk_averse_mpc(
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
    mode="trajwise",
    mpc_cls=RiskAverseMPC,
    cost_args={},
):
    """Create Risk Averse MPC

    Usage:
        ```
        controller, runner = build_risk_averse_mpc(...)
        # weights.shape nfeats * (nbatch, nrisk)
        actions = controller(state, weights=all_weights)
        ```

    """

    build_costs = partial(build_multi_costs, mode=mode)

    controller, runner = build_mpc(
        env=env,
        f_cost=f_cost,
        horizon=horizon,
        dt=dt,
        replan=replan,
        T=T,
        engine=engine,
        method=method,
        name=name,
        test_mode=test_mode,
        add_bias=add_bias,
        build_costs=build_costs,
        mpc_cls=mpc_cls,
        cost_args=cost_args,
        support_batch=True,
    )

    return controller, runner
