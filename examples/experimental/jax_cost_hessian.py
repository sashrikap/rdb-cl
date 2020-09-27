from functools import partial
from rdb.optim.utils import *
from rdb.optim.optimizer import *
from rdb.optim.runner import Runner
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import Profiler
from rdb.infer import *
from jax.lax import fori_loop, scan
from jax.ops import index_update
import jax.random as random
import jax.numpy as jnp
import numpy as onp
import time
import jax
import gym, rdb.envs.drive2d
from rdb.optim.mpc import build_forward, build_costs
from jax import jacfwd, jacrev


def test_runner():
    env_name = "Week6_02-v1"
    env = gym.make(env_name)
    env.reset()
    controller, runner = build_mpc(
        env,
        env.main_car.cost_runtime,
        dt=env.dt,
        name=env_name,
        horizon=10,
        replan=5,
        engine="jax",
        method="adam",
    )
    weights = {"dist_cars": 0.001, "dist_lanes": 10000.0}
    weights = DictList([weights])
    actions = controller(env.state, weights=weights)
    hessian, norm = runner.compute_hessian(env.state, actions, weights)
    print("Norm", norm)


def test_hessian():
    env_name = "Week6_02-v1"
    env = gym.make(env_name)
    env.reset()

    dt = 0.1
    horizon = 10
    f_dyn = env.dynamics_fn
    f_feat = env.features_fn
    f_cost = env.main_car.cost_runtime
    xdim, udim = env.xdim, env.udim
    h_traj = build_forward(f_dyn, xdim, udim, horizon, dt)
    h_costs = build_costs(udim, horizon, h_traj, f_cost)
    h_csum = lambda x0, us, weights: jnp.sum(h_costs(x0, us, weights))

    # Gradient w.r.t. x and u
    h_grad = jax.jit(jax.grad(h_csum, argnums=(0, 1)))
    h_grad_u = lambda x0, us, weights: h_grad(x0, us, weights)[1]

    weights = {"dist_cars": 0.001, "dist_lanes": 10000.0}
    weights = (
        DictList(weights, expand_dims=True).prepare(env.features_keys).numpy_array()
    )

    # Hessian w.r.t. u
    h_hess = jax.jit(jacfwd(jacrev(h_csum, argnums=1), argnums=1))
    h_hess2 = jacfwd(jacrev(h_csum, argnums=1), argnums=1)
    x0 = env.state
    us = jnp.zeros((1, horizon, env.udim))
    import pdb

    pdb.set_trace()
    h_costs(x0, us.swapaxes(0, 1), weights)
    H = h_hess(x0, us.swapaxes(0, 1), weights)
    h_hess(x0, jnp.ones((1, horizon, env.udim)).swapaxes(0, 1), weights)
    print(f"Hessian with shape", H.shape)


def exp_hessian():
    def predict(params, inputs):
        return jnp.sum(params["W"].dot(inputs) + params["b"])

    params = dict(W=np.ones((5, 2)), b=np.ones((5, 3)))
    data = jnp.ones((2, 3))
    J_dict = jacrev(predict)(params, data)
    for k, v in J_dict.items():
        print("Jacobian from {} to logits is".format(k))
        print(v)

    f = lambda W: predict({"W": W, "b": params["b"]}, data)

    def hessian(f):
        return jacfwd(jacrev(f))

    H = hessian(f)(params["W"])
    print("hessian, with shape", H.shape)
    print(H)


if __name__ == "__main__":
    test_runner()
    # test_hessian()
    # exp_hessian()
