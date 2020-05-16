from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "purple",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def read_seed(path):
    # print(path)
    data = np.load(path, allow_pickle=True).item()
    return data


def cleanup(arr, max_len):
    if len(arr) == 1:
        return np.array(arr)
    else:
        arrs = []
        for a in arr:
            if len(a) >= max_len + PADDING:
                arrs.append(a[PADDING : PADDING + max_len])
        return np.array(arrs)


def load_data(eval_plot_data, map_plot_data, exp_name, save_name=None):
    if save_name is None:
        save_name = exp_name

    map_seedpaths, map_data = [], []
    eval_seedpaths, eval_data = [], []
    # Load map_seed data
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for file in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            if save_name in file and "npy" in file and file.endswith("npy"):
                exp_path = os.path.join(exp_dir, exp_name, file)
                use_bools = [str(s) in exp_path for s in use_seeds]
                not_bools = [str(s) in exp_path for s in not_seeds]
                if os.path.isfile(exp_path) and "map" in file:
                    print(exp_path, use_seeds, not_bools)
                    if onp.any(use_bools) and not onp.any(not_bools):
                        map_seedpaths.append(exp_path)
                elif os.path.isfile(exp_path) and "map" not in file:
                    print(exp_path, use_seeds, not_bools)
                    if onp.any(use_bools) and not onp.any(not_bools):
                        eval_seedpaths.append(exp_path)

    if len(map_seedpaths) == 0:
        return

    map_data = [read_seed(path) for path in map_seedpaths]
    eval_data = [read_seed(path) for path in eval_seedpaths]

    for map_d, eval_d in zip(map_data, eval_data):
        for method in map_d.keys():
            map_hist = map_d[method]
            eval_hist = eval_d[method]
            if method not in eval_plot_data:
                eval_plot_data[method] = []
            if method not in map_plot_data:
                map_plot_data[method] = []
            for _idx in range(len(map_hist)):
                # for _idx in range(len(map_hist)):
                map_plot_data[method].append(
                    np.array(map_hist[_idx]["map_perform"]).mean()
                )
                eval_plot_data[method].append(eval_hist[_idx]["perform"])


def plot_data(eval_plot_data, map_plot_data):
    # Only look at first proposed task
    _, ax = plt.subplots(figsize=(10, 10))
    for method in eval_plot_data:
        ax.plot(
            -1 * np.array(eval_plot_data[method]),
            -1 * np.array(map_plot_data[method]),
            "o",
            color=colors[method],
            label=method,
            markersize=5,
        )

    # plt.xticks(x, list(data.keys()))
    plt.legend(loc="upper left")
    xy_range = (-0.5, 5)
    ax.plot(xy_range, xy_range, "k--", linewidth=1, color="gray")

    ax.set_xlim(*xy_range)
    ax.set_ylim(*xy_range)

    # ax.set_xlim(1e-1, 10)
    # ax.set_ylim(1e-1, 10)
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.set_ylabel("MAP Regret on Proposed task")
    ax.set_xlabel("MAP Regret")
    # plt.axis('scaled')
    # plt.show()
    ax.set_title(f"MAP regret on proposed task vs MAP regret")
    plt.savefig(os.path.join(exp_dir, exp_name, f"map_vs_eval.png"))


if __name__ == "__main__":
    N = -1
    all_seeds = [0, 1, 2, 3, 21, 22]
    not_seeds = []
    exp_dir = "data/200402"
    exp_name = "active_ird_ibeta_50_true_w1_eval_unif_128_difficult_seed_0_603_adam"
    alt_name = "active_ird_ibeta_50_true_w1_eval_unif_128_seed_0_603_adam"

    MAX_LEN = 4
    MAX_RANDOM_LEN = 4
    PADDING = 0

    use_seeds = [str(random.PRNGKey(si)) for si in all_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]

    eval_plot_data = {}
    map_plot_data = {}
    load_data(eval_plot_data, map_plot_data, exp_name)
    load_data(eval_plot_data, map_plot_data, exp_name, alt_name)
    plot_data(eval_plot_data, map_plot_data)
