from rdb.exps.active import ActiveInfoGain
from rdb.optim.mpc import build_mpc
from rdb.distrib.particles import *
from rdb.infer import *
import numpy as onp


def ird_controller_fn(env, name=""):
    controller, runner = build_mpc(
        env, env.main_car.cost_runtime, dt=env.dt, replan=False, name=name, horizon=10
    )
    return controller, runner


def env_fn():
    import gym, rdb.envs.drive2d

    env = gym.make("Week6_01-v1")
    env.reset()
    return env


env = env_fn()
weight_params = {"max_weights": 15, "bins": 200}
normalized_key = "dist_cars"
active_fn = ActiveInfoGain(
    rng_key=None, beta=10, weight_params=weight_params, debug=False
)


eval_server = ParticleServer(
    env_fn,
    ird_controller_fn,
    parallel=True,
    normalized_key=normalized_key,
    weight_params=weight_params,
    max_batch=10,
)
eval_server.register("Active", 2)


def build_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.features_keys:
            w[key] = onp.random.random()
        weights.append(w)
    return DictList(weights)


def build_particles(num_weights):
    controller, runner = ird_controller_fn(env)
    ps = Particles(
        rng_name="",
        rng_key=None,
        env_fn=None,
        env=env,
        controller=controller,
        runner=runner,
        normalized_key=normalized_key,
        save_name="test_particles",
        weights=build_weights(num_weights),
        save_dir=f"./test",
        weight_params=weight_params,
    )
    return ps


def run_speed():
    candidates = env.all_tasks[:72]
    belief = build_particles(10)
    obs = build_particles(1)
    scores = []
    eval_server.compute_tasks("Active", belief, candidates, verbose=True)
    eval_server.compute_tasks("Active", obs, candidates, verbose=True)

    for next_task in tqdm(candidates):
        scores.append(
            active_fn(
                onp.array([next_task]), belief, [obs], env.features_keys, verbose=False
            )
        )
    scores = onp.array(scores)


if __name__ == "__main__":
    run_speed()
