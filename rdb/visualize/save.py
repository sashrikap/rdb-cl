import numpy as np

"""
Basic savers & loaders for trajectories
"""


def save_rewards(xs, ys, rews, path="data.npz"):
    np.savez(path, xs, ys, rews)
