import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d

from tqdm import tqdm
from rdb.infer.utils import collect_trajs
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features

DUMMY_ACTION = False
DRAW_HEAT = False
REPLAN = False
MAKE_MP4 = False
# ENV_NAME = "Week3_02-v0"  # Highway
# TASK = (0.2, -0.7)
ENV_NAME = "Week6_01-v0"  # Blockway
# TASK = (0.2, -0.7, -0.1, 0.4)
TASK = (-0.20000005, -5.9604645e-08, -0.16, 0.19999993)
# TASK = (0.2, -0.7, 0.0, 0.4, -0.13, 0.3, -0.13, 0.5)

env = gym.make(ENV_NAME)
env.reset()
main_car = env.main_car
horizon = 10
# T = 30
T = 10
weights = {
    "dist_cars": 1.0,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 1.25,
    "speed": 0.05,
    "control": 0.1,
}

if not DUMMY_ACTION:
    if not REPLAN:
        T = horizon
    optimizer, runner = shooting_method(
        env, main_car.cost_runtime, horizon, env.dt, replan=REPLAN, T=T
    )
    env.set_task(TASK)
    env.reset()
    state = copy.deepcopy(env.state)
    t1 = time.time()
    actions = optimizer(state, weights=weights)
    traj, cost, info = runner(state, actions, weights=weights)
    # trajs = collect_trajs([weights], state, optimizer, runner)
    t_compile = time.time() - t1
    print(f"Compile time {t_compile:.3f}")
    print(f"Total cost {cost}")
    for k, v in info["violations"].items():
        print(f"Violations {k}: {v.sum()}")
    for k, v in info["feats_sum"].items():
        print(f"Feats sum {k}: {v:.3f}")
else:
    actions = np.zeros((T, env.udim))

N = 10
t1 = time.time()
for _ in tqdm(range(N), total=N):
    env.reset()
    acs_ = optimizer(env.state, weights=weights)
    # runner(env.state, acs_, weights=weights)
t_opt = time.time() - t1
print(f"Optimizer fps {N/t_opt:.3f}")

t1 = time.time()
for _ in tqdm(range(N), total=N):
    env.reset()
    runner(env.state, acs_, weights=weights)
t_run = time.time() - t1
print(f"Runner fps {N/t_run:.3f}")


env.reset()
env.render("human", draw_heat=DRAW_HEAT, weights=weights)

for t in range(T):
    env.step(actions[t])
    env.render("human", draw_heat=DRAW_HEAT, weights=weights)
    time.sleep(0.2)


if MAKE_MP4:
    pathname = f"data/run_highway.mp4"
    render_env(env, state, actions, 10, pathname)
