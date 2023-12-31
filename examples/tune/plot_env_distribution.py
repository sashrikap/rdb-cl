import os
import gym
import rdb.envs.drive2d
import matplotlib.pyplot as plt
import numpy as onp
import matplotlib
import seaborn as sns

matplotlib.rcParams["text.usetex"] = True
matplotlib.rc("font", family="serif", serif=["Palatino"])
sns.set(font="serif", font_scale=1.4)
sns.set_style(
    "white",
    {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.facecolor": "white",
        "lines.markeredgewidth": 1,
    },
)

ROOT_DIR = "data/env_distribution"


env3 = gym.make("Week6_03-v1")  # Two Blockway sparse
env3.reset()
env2 = gym.make("Week6_02-v1")  # Two Blockway dense
env2.reset()
tasks3 = env3.all_tasks
difficulties3 = env3.all_task_difficulties
tasks2 = env2.all_tasks
difficulties2 = env2.all_task_difficulties

array = onp.array
data = [
    {
        "infogain": [
            array([-1.0, 0.0, -0.16, -1.9, 0.1, 0.3]),
            array([0.5, 1.5, -0.16, 1.6, 0.1, 1.5]),
            array([0.5, -1.25, -0.16, 0.5, 0.1, 1.2]),
            array([0.5, 1.0, -0.16, 1.0, 0.1, -1.4]),
            array([0.75, 0.0, -0.16, 0.7, 0.1, 1.1]),
            array([1.0, 0.25, -0.16, 0.4, 0.1, 0.3]),
            array([1.0, 0.25, -0.16, -1.6, 0.1, -1.6]),
            array([1.0, 0.25, -0.16, 0.5, 0.1, 1.4]),
            array([1.0, 0.25, -0.16, -1.5, 0.1, 1.3]),
            array([0.75, 0.25, -0.16, -1.3, 0.1, -1.6]),
            array([1.75, 0.5, -0.16, -1.7, 0.1, -0.7]),
            array([0.5, -0.25, -0.16, -1.3, 0.1, 0.5]),
            array([-1.5, 0.5, -0.16, -1.4, 0.1, -1.7]),
            array([0.75, 0.0, -0.16, -1.1, 0.1, -0.6]),
        ],
        "random": [
            array([-1.0, 0.0, -0.16, -1.9, 0.1, 0.3]),
            array([1.0, -0.25, -0.16, -0.9, 0.1, 0.4]),
            array([-2.0, -0.75, -0.16, 0.4, 0.1, 0.4]),
            array([-1.5, 0.5, -0.16, -0.6, 0.1, 0.4]),
            array([1.25, 1.75, -0.16, -1.4, 0.1, -2.0]),
            array([-0.5, -0.25, -0.16, 1.2, 0.1, -1.8]),
            array([-1.5, 1.0, -0.16, -0.5, 0.1, -1.6]),
            array([-1.0, 1.0, -0.16, 1.1, 0.1, 1.0]),
            array([0.75, 0.25, -0.16, -1.5, 0.1, 1.9]),
            array([-0.75, -1.75, -0.16, 1.5, 0.1, 1.8]),
            array([1.25, -0.5, -0.16, -2.0, 0.1, 1.0]),
            array([1.0, 0.25, -0.16, 1.2, 0.1, -0.8]),
            array([-0.5, -1.25, -0.16, -1.9, 0.1, -1.4]),
            array([-1.75, 0.25, -0.16, -0.5, 0.1, -1.9]),
        ],
    },
    {
        "infogain": [
            array([1.5, -1.0, -0.16, -1.2, 0.1, 0.5]),
            array([-1.0, 0.5, -0.16, 1.1, 0.1, 1.3]),
            array([0.5, 0.5, -0.16, 0.6, 0.1, 0.5]),
            array([0.5, 0.0, -0.16, -1.6, 0.1, -0.9]),
            array([0.75, 0.0, -0.16, -1.0, 0.1, -1.2]),
            array([0.5, -0.25, -0.16, 0.7, 0.1, 0.4]),
            array([-0.75, 0.25, -0.16, 0.5, 0.1, 1.5]),
            array([0.5, 1.75, -0.16, 1.5, 0.1, 0.2]),
            array([0.5, 1.5, -0.16, 1.5, 0.1, 1.5]),
            array([0.75, 0.0, -0.16, -0.6, 0.1, -0.5]),
            array([0.5, -0.25, -0.16, -1.1, 0.1, -1.0]),
            array([0.75, 0.0, -0.16, -0.4, 0.1, -0.4]),
            array([0.5, -0.25, -0.16, -0.6, 0.1, 0.3]),
            array([0.5, -0.25, -0.16, -0.8, 0.1, 0.8]),
            array([0.5, -0.25, -0.16, -1.5, 0.1, -0.8]),
        ],
        "random": [
            array([1.5, -1.0, -0.16, -1.2, 0.1, 0.5]),
            array([-0.25, 1.5, -0.16, 0.7, 0.1, 1.4]),
            array([-1.25, -0.75, -0.16, -1.1, 0.1, 0.6]),
            array([-0.5, -2.0, -0.16, -1.1, 0.1, -1.6]),
            array([-1.25, -2.0, -0.16, 1.6, 0.1, 0.2]),
            array([0.75, -1.75, -0.16, -2.0, 0.1, -1.5]),
            array([0.5, 1.5, -0.16, 1.8, 0.1, -1.4]),
            array([-0.5, -0.25, -0.16, -1.4, 0.1, -1.2]),
            array([-1.5, -0.25, -0.16, 0.8, 0.1, 0.4]),
            array([1.75, 1.5, -0.16, 0.9, 0.1, 1.8]),
            array([1.5, -1.5, -0.16, 1.7, 0.1, 1.5]),
            array([1.75, 1.5, -0.16, -1.1, 0.1, 0.3]),
            array([-0.25, 1.5, -0.16, -1.4, 0.1, -1.2]),
            array([-0.5, -0.5, -0.16, -1.7, 0.1, -1.6]),
            array([-0.75, 1.25, -0.16, 0.6, 0.1, -0.5]),
        ],
    },
    {
        "infogain": [
            array([-1.5, -1.75, -0.16, -1.8, 0.1, 1.0]),
            array([-0.25, 0.25, -0.16, 1.1, 0.1, 0.7]),
            array([0.5, -2.0, -0.16, 1.0, 0.1, 0.4]),
            array([0.5, 0.0, -0.16, 1.0, 0.1, 1.9]),
            array([0.5, -0.25, -0.16, -0.9, 0.1, 1.6]),
            array([0.5, -0.25, -0.16, 0.6, 0.1, 0.5]),
            array([0.5, 0.0, -0.16, -0.8, 0.1, 0.5]),
            array([0.5, 0.25, -0.16, 1.1, 0.1, -1.1]),
            array([1.25, 0.5, -0.16, -1.2, 0.1, 0.4]),
            array([0.75, 0.0, -0.16, -0.8, 0.1, 1.8]),
            array([-1.0, 0.5, -0.16, -0.9, 0.1, 0.2]),
            array([0.5, -0.25, -0.16, -1.4, 0.1, -1.3]),
            array([0.5, -0.25, -0.16, -0.8, 0.1, -0.9]),
            array([0.5, -0.25, -0.16, -0.6, 0.1, -1.6]),
            array([-1.75, 0.5, -0.16, -1.8, 0.1, -1.4]),
        ],
        "random": [
            array([-1.5, -1.75, -0.16, -1.8, 0.1, 1.0]),
            array([-1.5, -1.0, -0.16, -2.0, 0.1, 1.4]),
            array([-1.75, 1.0, -0.16, 1.0, 0.1, 1.7]),
            array([-0.25, -0.75, -0.16, 0.6, 0.1, 1.1]),
            array([-2.0, 1.0, -0.16, 0.4, 0.1, -0.7]),
            array([-1.0, 0.0, -0.16, 1.7, 0.1, 1.2]),
            array([1.75, 0.5, -0.16, 1.7, 0.1, 0.9]),
            array([0.75, -0.25, -0.16, 0.4, 0.1, 1.2]),
            array([0.5, -1.25, -0.16, 0.2, 0.1, 1.3]),
            array([-1.0, 1.0, -0.16, 0.4, 0.1, 1.7]),
            array([0.75, -1.0, -0.16, -1.3, 0.1, 0.4]),
            array([0.5, 0.5, -0.16, 1.8, 0.1, -0.6]),
            array([-1.5, -0.25, -0.16, 0.3, 0.1, -1.4]),
            array([1.75, -1.25, -0.16, 1.8, 0.1, -0.4]),
            array([-0.5, -0.25, -0.16, -0.4, 0.1, -1.2]),
        ],
    },
    {
        "infogain": [
            array([-1.0, 1.5, -0.16, 1.2, 0.1, 1.2]),
            array([0.5, 0.25, -0.16, 0.6, 0.1, 1.1]),
            array([0.5, -0.25, -0.16, 0.6, 0.1, 0.3]),
            array([0.5, 0.0, -0.16, 0.6, 0.1, -1.6]),
            array([1.0, 0.25, -0.16, -0.5, 0.1, -0.6]),
            array([1.0, 0.25, -0.16, -1.2, 0.1, -1.7]),
            array([0.5, -0.25, -0.16, -1.0, 0.1, 1.8]),
            array([0.5, -0.25, -0.16, -1.9, 0.1, -1.4]),
            array([-0.75, 0.5, -0.16, 1.7, 0.1, -0.7]),
            array([0.5, 0.0, -0.16, -0.7, 0.1, 0.8]),
            array([0.75, 0.5, -0.16, -1.3, 0.1, -1.9]),
            array([0.5, -0.25, -0.16, 0.2, 0.1, 0.4]),
            array([0.75, 0.0, -0.16, 0.7, 0.1, 0.6]),
            array([1.0, 0.25, -0.16, 0.7, 0.1, 1.8]),
            array([0.5, 1.0, -0.16, 1.6, 0.1, -1.9]),
        ],
        "random": [
            array([-1.0, 1.5, -0.16, 1.2, 0.1, 1.2]),
            array([-1.5, -1.5, -0.16, -1.7, 0.1, -1.3]),
            array([-1.75, 0.0, -0.16, 1.0, 0.1, 0.4]),
            array([0.5, 1.25, -0.16, -0.5, 0.1, -0.9]),
            array([-1.75, -1.0, -0.16, 0.4, 0.1, 1.6]),
            array([-0.5, 0.75, -0.16, 1.7, 0.1, 1.8]),
            array([-2.0, -1.0, -0.16, 0.2, 0.1, -1.3]),
            array([1.25, -1.0, -0.16, -0.7, 0.1, 0.8]),
            array([0.75, 1.0, -0.16, 0.5, 0.1, 1.9]),
            array([0.5, -0.25, -0.16, -1.3, 0.1, -1.0]),
            array([-1.0, 1.25, -0.16, -0.8, 0.1, -0.4]),
            array([-1.5, 1.0, -0.16, -2.0, 0.1, -0.4]),
            array([-0.75, -1.0, -0.16, 1.8, 0.1, -0.5]),
            array([-1.0, -0.75, -0.16, 1.9, 0.1, -1.1]),
            array([0.75, 1.25, -0.16, 1.7, 0.1, -0.4]),
        ],
    },
    {
        "infogain": [
            array([0.5, -1.75, -0.16, 1.5, 0.1, 1.8]),
            array([0.5, 0.75, -0.16, 1.1, 0.1, 0.2]),
            array([0.5, -0.25, -0.16, 1.5, 0.1, 0.8]),
            array([1.0, 0.25, -0.16, -1.0, 0.1, 0.7]),
            array([0.5, -0.25, -0.16, -1.9, 0.1, -1.1]),
            array([0.5, -0.25, -0.16, 0.4, 0.1, -0.5]),
            array([0.5, -0.25, -0.16, 0.2, 0.1, 1.1]),
            array([0.75, 0.0, -0.16, -0.4, 0.1, -1.7]),
            array([-2.0, 0.5, -0.16, 1.9, 0.1, -2.0]),
            array([0.75, 0.0, -0.16, -0.4, 0.1, -0.7]),
            array([0.5, -0.25, -0.16, 1.6, 0.1, -1.0]),
            array([-1.5, 0.5, -0.16, -1.9, 0.1, -1.5]),
            array([0.5, -1.25, -0.16, 1.5, 0.1, 0.6]),
            array([0.5, -0.25, -0.16, -1.4, 0.1, -0.6]),
            array([0.5, -0.25, -0.16, -0.9, 0.1, 1.6]),
            array([1.0, 0.25, -0.16, -1.0, 0.1, -1.2]),
        ],
        "random": [
            array([0.5, -1.75, -0.16, 1.5, 0.1, 1.8]),
            array([-0.75, -0.75, -0.16, 1.3, 0.1, -2.0]),
            array([1.5, -1.0, -0.16, -0.5, 0.1, 0.6]),
            array([1.75, -1.0, -0.16, 1.7, 0.1, -1.9]),
            array([1.75, -1.75, -0.16, -2.0, 0.1, 1.6]),
            array([-1.25, -0.5, -0.16, 0.4, 0.1, 1.8]),
            array([1.0, -1.5, -0.16, 1.4, 0.1, 1.7]),
            array([-1.25, -1.75, -0.16, 0.3, 0.1, -1.9]),
            array([-1.75, -0.5, -0.16, 1.6, 0.1, -1.0]),
            array([1.0, 1.75, -0.16, 0.2, 0.1, 1.8]),
            array([0.75, -0.75, -0.16, 1.9, 0.1, 1.1]),
            array([-1.0, -1.5, -0.16, -1.5, 0.1, 1.7]),
            array([-1.25, -2.0, -0.16, -0.4, 0.1, 0.9]),
            array([1.25, -1.0, -0.16, 1.7, 0.1, 0.8]),
            array([-0.75, -1.0, -0.16, 1.8, 0.1, 0.9]),
            array([0.75, 0.0, -0.16, 1.5, 0.1, 1.9]),
        ],
    },
]

# import pdb; pdb.set_trace()
keys = ["random", "infogain"]
# keys = []
colors = ["gray", "darkorange"]
nbins = 80


print(f"Env 2 {len(tasks2)} tasks; env 3 {len(tasks3)} tasks")
fig, axs = plt.subplots(
    1 + len(keys),
    1,
    figsize=(12, 6),
    dpi=80,
    gridspec_kw={"height_ratios": [3] + [1] * len(keys)},
    sharex=True,
)
if len(keys) == 0:
    ax0 = axs
else:
    ax0 = axs[0]
n, bins3, patches3 = ax0.hist(
    difficulties3,
    density=True,
    facecolor="b",
    alpha=0.5,
    bins=nbins,
    histtype="stepfilled",
    label=f"Training ({int(len(tasks3) / 1000)}k)",
)
n, bins2, patches2 = ax0.hist(
    difficulties2,
    density=True,
    facecolor="r",
    alpha=0.5,
    bins=nbins,
    histtype="stepfilled",
    label=f"Test ({int(len(tasks2) / 1000)}k)",
)
ax0.set_yticklabels([])
ax0.set_xticklabels([])

# ax0.legend(loc="upper right")

for ki, (key, c) in enumerate(zip(keys, colors)):
    highlight_tasks = []
    for i in range(len(data)):
        highlight_tasks += data[i][key]
    highlight_diffs = env3._get_task_difficulties(highlight_tasks)
    highlight_ns = onp.digitize(highlight_diffs, bins3)
    highlight_xs = bins3[highlight_ns]
    # import pdb; pdb.set_trace()
    markers = ax0.plot(
        highlight_xs,
        onp.ones_like(highlight_xs) * 0.9 + 0.05 * ki,
        color=c,
        marker="x",
        linewidth=2,
        markersize=4,
        label=f"{key} proposals",
        linestyle="None",
    )
    markers[0].set_clip_on(False)

    (counts, bins) = onp.histogram(highlight_diffs, bins=nbins)
    factor = 0.2
    axs[1 + ki].hist(
        bins[:-1], bins, weights=factor * counts, facecolor=c, histtype="stepfilled"
    )
    axs[1 + ki].set_yticklabels([])
    axs[1 + ki].set_xticklabels([])


ax0.legend(loc="upper right")
# axs.set_title("Proposals")
ax0.set_xlabel("Density")
ax0.set_xlabel("Difficulty")
plt.savefig(f"{ROOT_DIR}/distribution.png")
# plt.show()
