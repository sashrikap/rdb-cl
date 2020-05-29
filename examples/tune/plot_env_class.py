import os
import gym
import rdb.envs.drive2d
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import numpy as np

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


# import pdb; pdb.set_trace()
# keys = ["random", "infogain"]
keys = []
colors = ["gray", "darkorange"]
nbins = 140


print(f"Env 2 {len(tasks2)} tasks; env 3 {len(tasks3)} tasks")
fig, axs = plt.subplots(
    figsize=(16, 6),
    dpi=80,
    gridspec_kw={"height_ratios": [3] + [1] * len(keys)},
    sharex=True,
)
axs.hist(
    difficulties3,
    density=True,
    facecolor="b",
    alpha=0.5,
    bins=nbins,
    histtype="stepfilled",
    label=f"Training Tasks",
)

axs.hist(
    difficulties2,
    density=True,
    facecolor="m",
    alpha=0.5,
    bins=nbins,
    histtype="stepfilled",
    label=f"Test Tasks",
)

# Hide the right and top spines
axs.spines["right"].set_visible(False)
axs.spines["top"].set_visible(False)

axs.legend(loc="upper center", ncol=2)
# axs.set_title("Proposals")
axs.set_ylabel("Density")
axs.set_xlabel("Difficulty")
plt.savefig(f"{ROOT_DIR}/all_distribution.png")
# plt.show()
