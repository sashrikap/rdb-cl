import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
import rdb.optim.open as opt_open


REOPTIMIZE = True

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

y0_idx, y1_idx = 1, 5
state = copy.deepcopy(env.state)
state[y0_idx] = 0.3
state[y1_idx] = -0.1
env.state = state

opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)), env.state)
r_max = -1 * c_min
env.render("human")
print(f"Rmax {r_max}")
print(opt_u)

T = 20
# T = horizon
total_cost = 0
for t in range(T):
    if REOPTIMIZE:
        action = opt_u[0]
    else:
        action = opt_u[t]
    cost = main_car.cost_fn(env.state, action)
    total_cost += cost
    env.step(action)
    env.render("human")

    if REOPTIMIZE:
        opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)), env.state)
    r_max = -1 * c_min
    time.sleep(0.2)
print(f"Rew {-1 * total_cost:.3f}")
