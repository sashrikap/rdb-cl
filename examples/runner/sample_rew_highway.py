import gym
import time, copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.mpc import shooting_optimizer
from rdb.optim.runner import Runner
from matplotlib import pyplot as plt
from rdb.visualize.render import render_env

MAKE_MP4 = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = horizon

optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, T=T
)
runner = Runner(env, main_car.cost_runtime, main_car.cost_fn)
weights = {
    "dist_cars": 50,
    "dist_lanes": 30.0,
    "dist_fences": 5000.0,
    "speed": 1000.0,
    "control": 20.0,
}

# Sampling parameters
NUM_SAMPLES = 100
weights_vals = onp.array(list(weights.values()))
sample_weights = weights_vals * onp.random.normal(
    onp.zeros_like(weights_vals), 1.0, (NUM_SAMPLES,) + weights_vals.shape
)

# Set initial state
y0_idx, y1_idx = 1, 5
state = copy.deepcopy(env.state)
state[y0_idx] = -0.5
state[y1_idx] = 0.05
env.state = state

# Optimize for control
actions = optimizer(env.state, weights=weights)
traj, cost, info = runner(env.state, actions)

rew = -1 * cost
env.render("human")
print(f"Total cost {cost}")

# Render actions
for t in range(T):
    env.step(actions[t])
    env.render("human")
    time.sleep(0.2)
