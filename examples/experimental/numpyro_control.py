import matplotlib.pyplot as plt
import jax.numpy as np
import numpyro.distributions as dist
import rdb.envs.drive2d
import numpy as onp
import numpyro
import time
import jax
import gym
import os
from numpyro.util import control_flow_prims_disabled, fori_loop, optional
from jax.scipy.special import logsumexp
from scipy.stats import gaussian_kde
from jax.interpreters import xla, ad
from numpyro.infer import MCMC, NUTS
from rdb.optim.mpc import build_mpc
from jax import abstract_arrays
from jax import lax, random
from rdb.infer import *


env = gym.make("Week6_02-v1")
env.reset()
main_car = env.main_car
nbatch = 5
horizon = 10
T = 10

optimizer, runner = build_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=False,
    T=T,
    engine="numpyro",
    method="adam",
)


def forward_one_step(prev_log_prob, curr_word, transition_log_prob, emission_log_prob):
    log_prob_tmp = np.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_log_prob[:, curr_word]
    return logsumexp(log_prob, axis=0)


untraceable_controller_p = jax.core.Primitive("untraceable_controller")
untraceable_controller_p.multiple_results = True
# ad.defjvp2(
#     untraceable_controller_p, None, lambda tangent, ans, key, a: np.ones_like(ans[0])
# )


def untraceable_controller(weights_arr):
    """Take in weights_arr and return control actions.
    """
    return untraceable_controller_p.bind(weights_arr)


def untraceable_controller_impl(weights_arr):
    # we don't want jax to trace this function: onp.array
    state = np.repeat(env.state, nbatch, axis=0)
    actions = optimizer(state, weights=None, weights_arr=weights_arr, batch=False)
    return (np.array(actions),)
    # return (np.array(weights_arr),)


untraceable_controller_p.def_impl(untraceable_controller_impl)
zero_actions = np.zeros((nbatch, T, env.udim))


def untraceable_controller_abst(weights_arr):
    # return (abstract_arrays.raise_to_shaped(zero_actions),)
    # return (abstract_arrays.ShapedArray(zero_actions.shape, zero_actions.dtype), )
    # return (abstract_arrays.raise_to_shaped(weights_arr),)
    return (abstract_arrays.ShapedArray(weights_arr.shape, weights_arr.dtype),)


untraceable_controller_p.def_abstract_eval(untraceable_controller_abst)
# ShapedArray(aval.shape, aval.dtype, weak_type=weak_type)

# xla.translations[untraceable_controller_p] = xla.lower_fun(
#     untraceable_controller_impl, instantiate=False
# )


def untraceable_controller_xla(c, xc):
    return c.Neg(xc)


xla.translations[untraceable_controller_p] = untraceable_controller_xla
# xla.backend_specific_translations['cpu'][untraceable_controller_p] = untraceable_controller_xla

# xla.backend_specific_translations['cpu'][untraceable_controller_p] = untraceable_controller_xla


# block gradient
ad.defvjp(untraceable_controller_p, lambda g, x: x * 0)
# ad.defvjp2(untraceable_controller_p, None, lambda g, ans, x: g * 0)


@jax.jit
def jax_sum(weights_arr, feats_arr):
    return np.mean(weights_arr * feats_arr)


def ird_experimental():
    dist_cars = np.exp(
        numpyro.sample("dist_cars", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_lanes = np.exp(
        numpyro.sample("dist_lanes", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_objects = np.exp(
        numpyro.sample("dist_objects", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed = np.exp(
        numpyro.sample("speed", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed_over = np.exp(
        numpyro.sample("speed_over", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed_under = np.exp(
        numpyro.sample("speed_under", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control = np.exp(
        numpyro.sample("control", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_thrust = np.exp(
        numpyro.sample("control_thrust", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_brake = np.exp(
        numpyro.sample("control_brake", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_turn = np.exp(
        numpyro.sample("control_turn", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_fences = np.exp(
        numpyro.sample("dist_fences", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    weights_arr = np.array(
        [
            dist_cars,
            dist_lanes,
            dist_objects,
            speed,
            speed_over,
            speed_under,
            control,
            control_thrust,
            control_brake,
            control_turn,
            dist_fences,
        ]
    )
    weights = {
        "dist_cars": dist_cars,
        "dist_lanes": dist_lanes,
        "speed": speed,
        "dist_objects": dist_objects,
        "speed_over": speed_over,
        "speed_under": speed_under,
        "control": control,
        "control_thrust": control_thrust,
        "control_turn": control_turn,
        "control_brake": control_brake,
        "dist_fences": dist_fences,
    }

    # weights_onp = {"dist_cars": onp.array(dist_cars)}

    state = np.repeat(env.state, nbatch, axis=0)

    # dummy compilation
    def cost_fn(weights_arr):
        actions = untraceable_controller(np.array(weights_arr))[0]
        print("actions", actions.shape)
        # traj, costs, info = runner(
        #     state, actions, weights=None, weights_arr=np.array(weights_arr), jax=True
        # )
        # weights_arr = DictList([weights], jax=True).numpy_array()
        # feats_sum = info["feats_sum"].numpy_array()
        # cost = costs.mean(axis=0)
        cost = actions.sum()
        return cost

    jax.jit(cost_fn)(weights_arr)
    cost_grad = jax.grad(cost_fn)
    # import pdb; pdb.set_trace()
    cost = cost_fn(weights_arr)
    # import pdb; pdb.set_trace()
    cost_g = cost_grad(weights_arr)
    import pdb

    pdb.set_trace()
    beta = 5
    log_prob = -1 * beta * cost
    numpyro.factor("forward_log_prob", log_prob)


def warmup():
    state = np.repeat(env.state, nbatch, axis=0)
    optimizer(state, weights=None, weights_arr=np.zeros((11, nbatch)))


def main():
    warmup()

    print("Starting inference...")
    rng_key = random.PRNGKey(2)
    kernel = NUTS(ird_experimental)
    num_warmup = 20
    num_samples = 40
    mcmc = MCMC(
        kernel,
        20,
        40,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    import pdb

    pdb.set_trace()
    print_results(samples)


if __name__ == "__main__":
    main()
