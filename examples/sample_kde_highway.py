import gym, time, copy
import numpy as onp
import jax.numpy as np
import rdb.envs.drive2d
import scipy.stats
from rdb.optim.open import shooting_optimizer
from functools import partial
from tqdm import tqdm

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm

# Numerical values used to help KDE function
NOISE_WEIGHT = 0.01
DIAG_WEIGHT = 0.001

weights1 = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 200.0,
    "speed": 4.0,
    "control": 80.0,
}
weights2 = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 200.0,
    "speed": 16.0,
    "control": 80.0,
}

env = gym.make("Week3_01-v0")
obs = env.reset()
main_car = env.main_car
udim = 2
horizon = 10

cost_runtime = main_car.cost_runtime
optimizer = shooting_optimizer(
    env.dynamics_fn, cost_runtime, udim, horizon, env.dt, mpc=True
)

y0_idx, y1_idx = 1, 5
y0_range = np.arange(-1.0, 1.01, 0.9)
y1_range = np.arange(-1.0, 1.01, 0.9)
state0 = copy.deepcopy(env.state)
state_dim = len(state0)

all_trajs1 = []
all_trajs2 = []
for y0 in tqdm(y0_range):
    trajs1 = []
    trajs2 = []
    for y1 in y1_range:
        state = copy.deepcopy(state0)
        state[y0_idx] = y0
        state[y1_idx] = y1
        opt_u1, c_min1, info1 = optimizer(
            np.zeros((horizon, udim)), state, weights=weights1
        )
        opt_u2, c_min2, info2 = optimizer(
            np.zeros((horizon, udim)), state, weights=weights2
        )
        trajs1.append(info1["xs"])
        trajs2.append(info2["xs"])
    all_trajs1.append(trajs1)
    all_trajs2.append(trajs2)

all_trajs1 = onp.array(all_trajs1)
all_trajs2 = onp.array(all_trajs2)

flat_trajs1 = all_trajs1.reshape(-1, (horizon + 1) * state_dim).T
flat_trajs2 = all_trajs2.reshape(-1, (horizon + 1) * state_dim).T
# small noise to prevent singularity
noise = onp.random.rand(*flat_trajs1.shape) * NOISE_WEIGHT

kde_fn = scipy.stats.gaussian_kde(flat_trajs1 + noise)
min_eig = min(onp.linalg.eigvalsh(kde_fn.covariance))
while min_eig < 0:
    kde_fn.covariance += onp.eye(flat_trajs1.shape[0]) * DIAG_WEIGHT
    min_eig = min(onp.linalg.eigvalsh(kde_fn.covariance))

all_dists = []
for trajs1, trajs2 in zip(all_trajs1, all_trajs2):
    dists = []
    for t1, t2 in zip(trajs1, trajs2):
        d = kde_fn(t2.flatten())
        dists.append(d)
    all_dists.append(dists)


fig = plt.figure()
ax = plt.axes(projection="3d")
X, Y = onp.meshgrid(y0_range, y1_range)
Z = onp.array(all_dists).T
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("Car 0 position")
ax.set_ylabel("Car 1 position")
ax.set_title("Reward")
plt.show()
