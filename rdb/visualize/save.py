import numpy as onp

"""
Basic savers & loaders for trajectories
"""


def save_rewards(xs, ys, rews, path="data.npz"):
    onp.savez(path, xs, ys, rews)
