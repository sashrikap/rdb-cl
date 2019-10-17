import gym
import time
import jax.numpy as np
import rdb.envs.drive2d
import rdb.optim.open as opt_open

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = main_car.horizon
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.reward_fn, udim, horizon, env.dt
)

now = time.time()
opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)), env.state)
# print(time.time() - now)

env.render("human")
time.sleep(1)
T = 20
for t in range(T):
    print("optimal u", opt_u[0])
    env.step(opt_u[0])
    print("render", env.main_car.state)
    env.render("human")
    opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)), env.state)
    time.sleep(1)
