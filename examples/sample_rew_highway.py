import gym
import time, copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from matplotlib import pyplot as plt
from rdb.visualize.render import render_env

REOPTIMIZE = True
MAKE_MP4 = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, mpc=REOPTIMIZE
)
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
opt_u, c_min, info = optimizer(np.zeros((horizon, udim)), env.state, weights=weights)
r_max = -1 * c_min
env.render("human")
print(f"Rmax {r_max}")
print(opt_u)

T = horizon
total_cost = 0
actions = []
for t in range(T):
    action = opt_u[t]
    actions.append(action)
    cost = main_car.cost_fn(env.state, action)
    total_cost += cost
    env.step(action)
    env.render("human")

    r_max = -1 * c_min
    time.sleep(0.2)

print(f"Rew {-1 * total_cost:.3f}")
