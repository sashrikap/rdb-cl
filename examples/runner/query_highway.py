import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.mpc import shooting_method
from rdb.optim.runner import Runner

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
y0_idx, y1_idx = 1, 5

# Scenario 1
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 200.0,
    "speed": 4.0,
    "control": 80.0,
}
T = 20
REPLAN = True
optimizer, runner = shooting_method(
    env, main_car.cost_runtime, udim, horizon, env.dt, T=T, replan=REPLAN
)
state = copy.deepcopy(env.state)
state[y0_idx] = 0.4
state[y1_idx] = -0.2
env.state = state

actions = optimizer(env.state, weights=weights)
traj, cost, info = runner(state, actions)
env.render("human")
print(f"Total cost {cost}")

total_cost = 0
for t in range(T):
    env.step(actions[t])
    env.render("human")
    time.sleep(0.2)
