import gym
import time
import numpy as onp
import rdb.envs.drive2d

env = gym.make("Week3_02-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = main_car.horizon

now = time.time()
opt_u = onp.zeros((horizon, udim))
opt_u[:, 0] = 0.0
opt_u[:, 1] = 2

weights = {
    "dist_cars": 10.0,
    "dist_lanes": 0.0,
    "dist_fences": 0.0,
    "speed": 1000.0,
    "control": 20.0,
}

env.render("human")
time.sleep(2)
for t in range(horizon):
    env.step(opt_u[t])
    env.render("human", draw_heat=True, weights=weights)
    time.sleep(1)
