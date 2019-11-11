import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner

# Environment setup
env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 10
y0_idx, y1_idx = 1, 5
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, T=T
)
runner = Runner(env, main_car)
state_const = copy.deepcopy(env.state)

# Scenario 1
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 200.0,
    "speed": 4.0,
    "control": 80.0,
}
# Training environments
train_pairs = [(0.4, -0.2), (0.2, -0.4)]

for idx, (y0, y1) in enumerate(train_pairs):
    state = copy.deepcopy(state_const)
    state[y0_idx] = y0
    state[y1_idx] = y1
    env.state = state

    actions = optimizer(state, weights=weights)
    traj, cost, info = runner(state, actions)
    print(f"Total cost: {cost:.3f}")
    text = f"Total cost: {cost:.3f}\nTraining [{idx}]"
    total_cost = 0
    for t in range(T):
        env.step(actions[t])
        env.render("human", text=text)
        time.sleep(0.2)
