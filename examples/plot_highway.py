import gym
import time
import copy
import jax.numpy as np
import numpy as onp
import rdb.envs.drive2d
import rdb.optim.open as opt_open
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm


env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 20
opt_u_fn = opt_open.optimize_u_fn(
    env.dynamics_fn, main_car.reward_fn, udim, horizon, env.dt
)

now = time.time()
state0 = copy.deepcopy(env.state)
y0_idx = 1
y1_idx = 5


opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)), env.state)
y0_range = np.arange(-2, 2, 0.05)
y1_range = np.arange(-2, 2, 0.05)

all_rews = []
for y0 in tqdm(y0_range):
    rews = []
    for y1 in y1_range:
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u, r_max, info = opt_u_fn(np.zeros((horizon, udim)), state)
        rews.append(r_max)
    all_rews.append(rews)


fig = plt.figure()
ax = plt.axes(projection="3d")
X, Y = onp.meshgrid(y0_range, y1_range)
Z = onp.array(all_rews)
# ax.plot_wireframe(X, Y, Z, color='black')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Car 0 position")
ax.set_ylabel("Car 1 position")
ax.set_title("Reward")
plt.show()
