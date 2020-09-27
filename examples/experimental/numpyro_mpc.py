import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpyro.distributions as dist
import rdb.envs.drive2d
import numpy as onp
import numpyro
import time
import jax
import gym
import os
from numpyro.util import control_flow_prims_disabled, fori_loop, optional
from numpyro.handlers import scale, condition, seed
from jax.scipy.special import logsumexp
from scipy.stats import gaussian_kde
from jax.interpreters import xla, ad
from numpyro.infer import MCMC, NUTS
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import Profiler
from jax import abstract_arrays
from jax import lax, random
from rdb.infer import *


def env_fn():
    env = gym.make("Week6_02-v1")
    env.reset()
    return env


env = env_fn()
main_car = env.main_car
nbatch = 5
horizon = 10
T = 10

controller, runner = build_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=False,
    T=T,
    engine="numpyro",
    method="adam",
)


def ird_experimental(feats_sum):
    t1 = time()
    dist_cars = jnp.exp(
        numpyro.sample("dist_cars", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_lanes = jnp.exp(
        numpyro.sample("dist_lanes", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_objects = jnp.exp(
        numpyro.sample("dist_objects", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed = jnp.exp(
        numpyro.sample("speed", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed_over = jnp.exp(
        numpyro.sample("speed_over", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    speed_under = jnp.exp(
        numpyro.sample("speed_under", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control = jnp.exp(
        numpyro.sample("control", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_thrust = jnp.exp(
        numpyro.sample("control_thrust", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_brake = jnp.exp(
        numpyro.sample("control_brake", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    control_turn = jnp.exp(
        numpyro.sample("control_turn", dist.Uniform(-10, 10), sample_shape=(nbatch,))
    )
    dist_fences = jnp.exp(
        numpyro.sample("dist_fences", dist.Uniform(-10, 10), sample_shape=(nbatch,))
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
    weights = DictList(
        {
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
        },
        jax=True,
    )
    costs = weights * feats_sum
    beta = 5
    log_prob = -1 * beta * costs.numpy_array().mean(axis=0).sum()
    numpyro.factor("forward_log_prob", log_prob)
    print(f"experimental {(time() - t1):.3f}")
    return weights_arr


def warmup():
    state = jnp.repeat(env.state, nbatch, axis=0)
    controller(state, weights=None, weights_arr=np.zeros((11, nbatch)))


def build_feats_sum(rng_key):
    weights = {
        "dist_cars": 5,
        "dist_lanes": 5,
        "dist_fences": 0.35,
        "dist_objects": 10.25,
        "speed": 5,
        "control": 0.1,
    }
    tasks = env.all_tasks[:nbatch]
    ws = DictList([weights] * nbatch)
    ps = Particles(
        rng_key=rng_key,
        env_fn=env_fn,
        controller=controller,
        runner=runner,
        weights=ws,
        save_name="",
        normalized_key="dist_cars",
        weight_params={},
        fig_dir=None,
        save_dir=None,
        env=env,
    )
    feats_sum = ps.get_features_sum(tasks)[np.diag_indices(nbatch)]
    return feats_sum


def main_numpyro():
    warmup()

    rng_key = random.PRNGKey(2)
    kernel = NUTS(ird_experimental)
    num_warmup = 20
    num_samples = 40
    mcmc = MCMC(kernel, 20, 40, progress_bar=True)

    feats_sum = build_feats_sum(rng_key)
    mcmc.run(rng_key, feats_sum)
    samples = mcmc.get_samples()
    print_results(samples)


def main_jit():
    warmup()
    rng_key = random.PRNGKey(2)
    kernel = seed(ird_experimental, rng_key)
    jit_kernel = jax.jit(kernel, static_argnums=(0,))
    feats_sum = build_feats_sum(rng_key)
    import pdb

    pdb.set_trace()
    jit_kernel(feats_sum)


if __name__ == "__main__":
    # main_numpyro()
    main_jit()
