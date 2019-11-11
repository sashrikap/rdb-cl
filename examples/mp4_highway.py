import gym
import time, copy
import jax.numpy as np
import rdb.envs.drive2d
from rdb.optim.open import shooting_optimizer
from rdb.visualize.render import render_env

REPLAN = True
env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10
state = copy.deepcopy(env.state)

optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_fn, udim, horizon, env.dt, replan=REPLAN
)

actions = optimizer(env.state)
pathname = f"data/video.mp4"
render_env(env, state, actions, 10, pathname)
