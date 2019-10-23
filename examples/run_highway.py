import gym
import time
import jax.numpy as np
import rdb.envs.drive2d
import rdb.optim.open as opt_open

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

now = time.time()
opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, env.state)
r_max = -1 * c_min
# print(time.time() - now)

env.render("human")
time.sleep(1)
print(f"Rmax {r_max}")
print(opt_u)
# T = 20
T = horizon
total_cost = 0
for t in range(T):
    # action = opt_u[t]
    action = opt_u[0]
    cost = main_car.cost_fn(env.state, action)
    total_cost += cost
    env.step(action)
    env.render("human")
    opt_u, c_min, info = opt_u_fn(np.zeros((horizon, udim)) * 0.1, env.state)
    r_max = -1 * c_min
    time.sleep(0.2)
print(f"Rew {-1 * total_cost:.3f}")
