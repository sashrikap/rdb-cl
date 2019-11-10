import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer


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
if not REOPTIMIZE:
    T = horizon
optimizer = shooting_optimizer(
    env.dynamics_fn, env.cost_runtime, udim, horizon, env.dt, T=T, mpc=REOPTIMIZE
)
state = copy.deepcopy(env.state)
state[y0_idx] = 0.4
state[y1_idx] = -0.2
env.state = state

opt_u, c_min, info = optimizer(
    np.ones((horizon, udim)) * 0.0, env.state, weights=weights
)
r_max = -1 * c_min
env.render("human")
print(f"Rmax {r_max}")
print(opt_u)

total_cost = 0
for t in range(T):
    action = opt_u[t]
    cost = main_car.cost_fn(env.state, action)
    total_cost += cost
    env.step(action)
    env.render("human")

    r_max = -1 * c_min
    time.sleep(0.2)


# Scenario 2
