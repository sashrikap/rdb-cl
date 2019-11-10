import gym
import time
import numpy as onp
import rdb.envs.drive2d
import rdb.optim.open as opt_open

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = main_car.horizon

now = time.time()
opt_u = onp.zeros((horizon, udim))
opt_u[:, 0] = 0.0
opt_u[:, 1] = 2

env.render("human")
time.sleep(2)
for t in range(horizon):
    env.step(opt_u[t])
    env.render("human")
    time.sleep(1)
