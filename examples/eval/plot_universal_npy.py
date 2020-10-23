from jax import random
import os
import yaml
import jax.numpy as jnp
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rc("font", family="serif", serif=["Palatino"])
sns.set(font="serif", font_scale=2)
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


def read_seed(path):
    # print(path)
    if path.endswith("npy"):
        data = jnp.load(path, allow_pickle=True).item()
    elif path.endswith("npz"):
        data = jnp.load(path, allow_pickle=True)
    return data


colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "cornflowerblue",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def load_data(key="ird"):
    npypaths = []
    npydata = []
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for file in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            if "npy" in file and key in file and file.endswith("npy"):
                exp_path = os.path.join(exp_dir, exp_name, file)
                # print(exp_path)
                if os.path.isfile(exp_path):
                    use_bools = [str(s) in exp_path for s in use_seeds]
                    not_bools = [str(s) in exp_path for s in not_seeds]
                    # print(file, use_bools, not_bools)
                    if onp.any(use_bools) and not onp.any(not_bools):
                        npypaths.append(exp_path)

    for exp_path in npypaths:
        npydata.append(read_seed(exp_path))

    data = {}
    for idx, (npy_data, npy_path) in enumerate(zip(npydata, npypaths)):
        for beta, eval_hist in npy_data.items():
            if beta not in data.keys():
                data[beta] = {
                    "perform": [],
                    "all_perform": [],
                    "rel_perform": [],
                    "normalized_perform": [],
                    "all_normalized_performs": [],
                    "violation": [],
                    "all_violation": [],
                    "rel_violation": [],
                    "feats_violation": [],
                }
            data[beta]["perform"].append(eval_hist["perform"])
            if "all_perform" in eval_hist.keys():
                data[beta]["all_perform"].append(onp.array(eval_hist["all_perform"]))
                data[beta]["rel_perform"].append(onp.array(eval_hist["rel_perform"]))
                # data[beta]["all_normalized_performs"].append(onp.array(eval_hist["all_normalized_performs"]))
            data[beta]["normalized_perform"].append(eval_hist["normalized_perform"])
            data[beta]["violation"].append(eval_hist["violation"])
            data[beta]["all_violation"].append(eval_hist["all_violation"])
            data[beta]["rel_violation"].append(eval_hist["rel_violation"])
            data[beta]["feats_violation"].append(eval_hist["feats_violation"])
    return data


def print_perform(data, metric):
    for beta, b_data in data.items():
        print(
            f"{beta}: {onp.mean(b_data[metric]):.4f} std: {onp.std(b_data[metric]):.4f} ({len(b_data[metric])})"
        )


def plot_data():
    data = load_data(key="ird")
    print_perform(data, metric="normalized_perform")
    # print_perform(data, metric="perform")


if __name__ == "__main__":
    N = -1
    # use_seeds = [0, 1, 2, 3, 4]
    use_seeds = [0, 1, 2, 3, 4]
    not_seeds = []
    PADDING = 0
    SEPARATE_SAVE = False

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]

    exp_dir = "data/200927"
    exp_name = "universal_joint_init1v3_2000_uni_eval_risk"
    plot_data()
