import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

npzfile = np.load("grid.npz")
y0_range = npzfile["arr_0"]
y1_range = npzfile["arr_1"]
all_rews = npzfile["arr_2"]


def find_nearest_idx(a, a0):
    idx = np.abs(a - a0).argmin()
    return idx


def drawSphere(xCenter, yCenter, zCenter, r):
    # draw sphere
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    return (x, y, z)


bad_pts = [[0.8, 0.2]]


fig = plt.figure()
all_rews = np.array(all_rews)
ax = plt.axes(projection="3d")
X, Y = np.meshgrid(y0_range, y1_range)
Z = np.array(all_rews).T
surf = ax.plot_surface(
    X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
)
fig.colorbar(surf, shrink=0.5, aspect=5)

bad_pts = np.array(bad_pts)
bad_xs = np.array([find_nearest_idx(y0_range, pt[0]) for pt in bad_pts])
bad_ys = np.array([find_nearest_idx(y1_range, pt[1]) for pt in bad_pts])
bad_zs = np.array([Z[x, y] for (x, y) in zip(bad_xs, bad_ys)])
# ax.scatter(bad_pts[:, 0], bad_pts[:, 1], bad_zs - 10, c='r', marker='o', s=100)
for xi, yi, zi in zip(bad_pts[:, 0], bad_pts[:, 1], bad_zs):
    xs, ys, zs = drawSphere(xi, yi, zi, 0.1)
    ax.plot_wireframe(xs, ys, zs, color="r")

ax.set_xlabel("Car 0 position")
ax.set_ylabel("Car 1 position")
ax.set_title("Reward")
plt.show()
