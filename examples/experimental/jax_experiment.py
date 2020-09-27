import os
import time

import matplotlib.pyplot as plt
import numpy as onp
from scipy.stats import gaussian_kde
import jax
from jax import lax, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as onp
import gym
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import rdb.envs.drive2d
from rdb.optim.mpc import build_mpc
from rdb.infer import *


env = gym.make("Week6_02-v1")
env.reset()
main_car = env.main_car
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
    log_prob_tmp = jnp.expand_dims(prev_log_prob, axis=1) + transition_log_prob
    log_prob = log_prob_tmp + emission_log_prob[:, curr_word]
    return logsumexp(log_prob, axis=0)


@jax.jit
def jax_sum(weights_arr, feats_arr):
    return jnp.mean(weights_arr * feats_arr)


def ird_experimental():
    dist_cars = jnp.exp(
        numpyro.sample("dist_cars", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    dist_lanes = jnp.exp(
        numpyro.sample("dist_lanes", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    dist_objects = jnp.exp(
        numpyro.sample("dist_objects", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    speed = jnp.exp(numpyro.sample("speed", dist.Uniform(-10, 10), sample_shape=(5,)))
    speed_over = jnp.exp(
        numpyro.sample("speed_over", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    speed_under = jnp.exp(
        numpyro.sample("speed_under", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    control = jnp.exp(
        numpyro.sample("control", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    control_thrust = jnp.exp(
        numpyro.sample("control_thrust", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    control_brake = jnp.exp(
        numpyro.sample("control_brake", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    control_turn = jnp.exp(
        numpyro.sample("control_turn", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    dist_fences = jnp.exp(
        numpyro.sample("dist_fences", dist.Uniform(-10, 10), sample_shape=(5,))
    )
    weights_arr = jnp.array(
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

    state = jnp.repeat(env.state, 5, axis=0)

    # dummy compilation
    # optimizer(onp.zeros(state.shape), weights=None, weights_arr=, batch=False)

    try:
        actions = optimizer(
            state, weights=None, weights_arr=onp.array(weights_arr), batch=False
        )
    except:
        import pdb

        pdb.set_trace()
    traj, cost, info = runner(
        state, actions, weights=None, weights_arr=onp.array(weights_arr), batch=False
    )
    # import pdb; pdb.set_trace()
    weights_arr = DictList([weights], jax=True).numpy_array()
    feats_sum = info["feats_sum"].numpy_array()
    # cost = DictList([weights], jax=True).numpy_array() * info["feats_sum"].numpy_array()
    # cost = dist_cars * info["feats_sum"].onp_array()
    # cost = dist_cars * jnp.ones((1, 4))
    # cost = DictList([weights], jax=True).numpy_array() * jnp.array([[10]])
    # cost = dist_cars * jnp.array([[10]])
    cost = cost.mean(axis=0)
    cost = jax_sum(weights_arr, feats_sum)
    beta = 5
    log_prob = -1 * beta * cost
    numpyro.factor("forward_log_prob", log_prob)


def main():
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
