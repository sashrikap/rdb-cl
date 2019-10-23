import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
import rdb.optim.open as opt_open
from rdb.visualize.render import render_env

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
state0 = copy.deepcopy(env.state)

opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt
)

now = time.time()
opt_u, c_min, info = opt_u_fn(np.ones((horizon, udim)) * 0.1, env.state)
print(opt_u)
r_max = -1 * c_min

pathname = (
    f"data/191022/car 0 {state0[1]:.3f} car 1 {state0[5]:.3f} rmax {r_max:.3f}.mp4"
)
print(pathname)
render_env(env, state0, opt_u, 10, pathname)
