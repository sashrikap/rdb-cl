import gym
import copy
import jax.numpy as jnp
import numpy as onp
import rdb.envs.drive2d

from time import time, sleep
from tqdm import tqdm, trange
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DUMMY_ACTION = False
REPLAN = False
BENCHMARK = 500
BENCHMARK_SINGLE = True
BENCHMARK_BATCH = False
# ENGINE = "scipy"
# METHOD = "lbfgs"
ENGINE = "jax"
METHOD = "adam"
ENV_NAME = "Week6_02-v1"  # Two Blockway
TASK = "RANDOM"

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
T = 10
weights = {
    "dist_cars": 1.1,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 100.25,
    "speed": 0.05,
    "control": 0.1,
}
# weights = {"dist_lanes": 5.35}
if TASK == "RANDOM":
    num_tasks = len(env.all_tasks)
    print(f"Total tasks {num_tasks}")
    TASK = env.all_tasks[onp.random.randint(0, num_tasks)]
    # import pdb; pdb.set_trace()
env.set_task(TASK)
env.reset()

print(f"Task {TASK}")

if not REPLAN:
    T = horizon
optimizer, runner = build_mpc(
    env,
    main_car.cost_runtime,
    horizon,
    env.dt,
    replan=REPLAN,
    T=T,
    engine=ENGINE,
    method=METHOD,
)
state = copy.deepcopy(env.state)
t1 = time.time()
actions = optimizer(state, weights=weights, batch=False)
traj, cost, info = runner(state, actions, weights=weights, batch=False)
print(f"Engine {ENGINE}, method {METHOD}")
if BENCHMARK > 0:
    t_compile = time.time() - t1
    print(f"Compile time {t_compile:.3f}")
    print(f"Total cost {cost}")
    for k, v in info["violations"].items():
        print(f"Violations {k}: {v.sum()}")
    for k, v in info["feats_sum"].items():
        print(f"Feats sum {k}: {v.sum():.3f}")

    N = BENCHMARK

    if BENCHMARK_SINGLE:
        t1 = time.time()
        for _ in tqdm(range(N), total=N):
            env.reset()
            acs_ = optimizer(state, weights=weights, batch=False)
        t_opt = time.time() - t1
        print(f"Optimizer fps {N/t_opt:.3f}")

        t1 = time.time()
        for _ in tqdm(range(N), total=N):
            env.reset()
            runner(state, acs_, weights=weights, batch=False)
        t_run = time.time() - t1
        print(f"Runner fps {N/t_run:.3f}")

        t1 = time.time()
        for _ in tqdm(range(N), total=N):
            env.reset()
            for key in weights.keys():
                weights[key] = onp.random.random()
            acs_ = optimizer(state, weights=weights, batch=False)
        t_opt = time.time() - t1
        print(f"Optimizer (Changing weights) fps {N/t_opt:.3f}")

        t1 = time.time()
        for _ in tqdm(range(N), total=N):
            env.reset()
            for key in weights.keys():
                weights[key] = onp.random.random()
            runner(state, acs_, weights=weights, batch=False)
        t_run = time.time() - t1
        print(f"Runner (Changing weights) fps {N/t_run:.3f}")

    if BENCHMARK_BATCH:
        N_batch = 20
        optimizer2, runner2 = build_mpc(
            env,
            main_car.cost_runtime,
            horizon,
            env.dt,
            replan=REPLAN,
            T=T,
            engine=ENGINE,
            method=METHOD,
        )
        all_weights = []
        for _ in range(N):
            all_weights.append(weights)
        ws = DictList(all_weights)
        # re-compile once
        state_N = state.repeat(N, axis=0)
        acs = optimizer2(state_N, weights=ws)
        runner2(state_N, acs, weights=ws)
        t1 = time.time()
        for _ in trange(N_batch):
            acs = optimizer2(state_N, weights=ws)
        t_opt = time.time() - t1
        print(f"Optimizer (batch) fps {N_batch * N/t_opt:.3f}")
        t1 = time.time()
        for _ in trange(N_batch):
            runner2(state_N, acs, weights=ws)
        t_run = time.time() - t1
        print(f"Runner (batch) fps {N * N_batch/t_run:.3f}")

        t1 = time.time()
        all_weights = []
        for _ in range(N // 2):
            all_weights.append(weights)
        ws = DictList(all_weights)
        state_N = state.repeat(N // 2, axis=0)
        acs = optimizer2(state_N, weights=ws)
        runner2(state_N, acs, weights=ws)
