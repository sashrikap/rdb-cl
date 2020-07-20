"""Risk-Averse Model Predictive controllers.

Extends:
    * rdb.optim.mpc.py

Includes:
    * Trajectory-wise risk averse planning.
    * Step-wise risk averse planning

"""
import jax
import jax.numpy as np
from jax.lax import map as jmap
from jax.scipy.special import logsumexp
from functools import partial
from rdb.optim.mpc import build_mpc


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
        costs = np.array(costs)
        return np.exp(costs - logsumexp(costs, axis=axis))

    def _hard_max_ratio(costs, axis=None):
        costs = np.array(costs)
        idx = np.argmax(costs, axis=axis)
        ratio = np.zeros_like(costs)
        ratio[idx] = 1.0
        return ratio

    if softmax:
        _max_ratio = _soft_max_ratio
    else:
        _max_ratio = _hard_max_ratio

    def roll_cost(weights, x, us):
        vf_cost = jax.vmap(partial(f_cost, weights=weights))
        xs = roll_forward(x, us)
        costs = vf_cost(xs, us)
        return np.array(costs)

    @jax.jit
    def roll_stepwise_costs(x, us, all_weights):
        """Calculate step-wise trajectory costs

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            all_weights (ndarray): (nfeats, nweights)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        assert len(all_weights.shape) == 2
        #  shape: (nweights, horizon, nbatch)
        proll_cost = partial(roll_cost, x=x, us=us)
        #  shape (nweights, horizon, 1)
        all_weights = all_weights.swapaxes(0, 1)[:, :, None]
        proll_cost(weights=all_weights[0])
        #  shape (nweights, horizon, 1)
        costs = np.array(jmap(proll_cost, all_weights))
        assert len(costs.shape) == 3
        ratio = _max_ratio(costs, axis=0)
        return np.sum(ratio * costs, axis=0)

    @jax.jit
    def roll_trajwise_costs(x, us, all_weights):
        """Calculate trajectory features

        Args:
            x (ndarray): (nbatch, xdim,)
            us (ndarray): (horizon, nbatch, xdim)
            all_weights (ndarray): (nfeats, nweights)

        Output:
            cost (ndarray): (horizon, nbatch)

        """
        assert len(all_weights.shape) == 2
        proll_cost = partial(roll_cost, x=x, us=us)
        #  shape: (nweights, horizon, 1)
        all_weights = all_weights.swapaxes(0, 1)[:, :, None]
        #  shape (nweights, horizon, 1)
        costs = np.array(jmap(proll_cost, all_weights))
        assert len(costs.shape) == 3
        csums = np.sum(costs, axis=(1, 2))
        ratio = _max_ratio(csums, axis=0)
        return np.sum(ratio * costs, axis=0)

    if mode == "stepwise":
        return roll_stepwise_costs
    elif mode == "trajwise":
        return roll_trajwise_costs
    else:
        raise NotImplementedError


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
    cost_args={},
):
    return build_mpc(
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
        build_costs=build_multi_costs,
        cost_args=cost_args,
    )
