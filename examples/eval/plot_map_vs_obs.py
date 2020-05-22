from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import nonzero

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
    data = onp.load(path, allow_pickle=True).item()
    return data


def cleanup(arr, max_len):
    if len(arr) == 1:
        return onp.array(arr)
    else:
        arrs = []
        for a in arr:
            if len(a) >= max_len + PADDING:
                arrs.append(a[PADDING : PADDING + max_len])
        return onp.array(arrs)


def load_data(eval_plot_data, map_plot_data, obs_plot_data, exp_name, save_name=None):
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
            if method in not_method:
                continue
            map_hist = map_d[method]
            eval_hist = eval_d[method]
            if method not in eval_plot_data:
                eval_plot_data[method] = []
            if method not in map_plot_data:
                map_plot_data[method] = []
            if method not in obs_plot_data:
                obs_plot_data[method] = []
            for _idx in range(len(map_hist)):
                # for _idx in range(len(map_hist)):
                map_plot_data[method].append(
                    onp.array(map_hist[_idx]["belief_perform_relative"]).mean()
                )
                obs_plot_data[method].append(
                    onp.array(map_hist[_idx]["obs_perform_relative"]).mean()
                )
                eval_plot_data[method].append(eval_hist[_idx]["perform"])


def get_bin_mean(a, b_start, b_end):
    ind_upper = nonzero(a >= b_start)[0]
    a_upper = a[ind_upper]
    a_range = a_upper[nonzero(a_upper < b_end)[0]]
    mean_val = onp.mean(a_range)
    return mean_val


colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "purple",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def plot_data(eval_plot_data, map_plot_data, obs_plot_data):
    # Only look at first proposed task
    _, ax = plt.subplots(figsize=(10, 10))

    n = 0
    for method in eval_plot_data:
        ax.plot(
            -1 * onp.array(eval_plot_data[method]),
            -1 * onp.array(map_plot_data[method]),
            "o",
            color=colors[method],
            label=method,
            markersize=5,
        )

    # plt.xticks(x, list(data.keys()))
    plt.legend(loc="upper left")
    xy_range = (-0.5, 5)
    ax.plot(xy_range, xy_range, "k--", linewidth=1, color="gray")

    # ax.set_xlim(*xy_range)
    # ax.set_ylim(*xy_range)

    # ax.set_xlim(1e-1, 10)
    # ax.set_ylim(1e-1, 10)
    # ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.set_ylabel("Posterior Regret on Proposed task")
    ax.set_xlabel("Current Posterior Regret")
    # plt.axis('scaled')
    ax.set_title(
        f"Posterior regret on next task vs Current posterior regret (higher is better)"
    )
    plt.savefig(os.path.join(exp_dir, exp_name, f"map_vs_eval.png"))
    plt.show()


def plot_binned_data(eval_plot_data, map_plot_data, obs_plot_data, plot_obs=False):
    # Only look at first proposed task
    _, ax = plt.subplots(figsize=(10, 10))

    bins = [-5, 2, 5]
    labels = []
    for n in range(0, len(bins) - 1):
        if n == 0:
            labels.append(f"regret<={onp.exp(bins[n+1]):.02f}")
        elif n == len(bins) - 2:
            labels.append(f"regret>{onp.exp(bins[n]):.02f}")
        else:
            labels.append(f"{onp.exp(bins[n]):.02f}< regret<={onp.exp(bins[n+1]):.02f}")
    print(labels)
    n = 0
    binned_data = {}
    for method in eval_plot_data:
        log_eval_data = onp.log(-1 * onp.array(eval_plot_data[method]))
        map_arr = onp.array(map_plot_data[method])
        map_arr[map_arr > 0] = -1e-3
        # log_map_data = onp.log(-1 * onp.array(map_arr))
        # log_obs_data = onp.log(-1 * onp.array(obs_plot_data[method]))
        map_data = -1 * onp.array(map_arr)

        binned_data[method] = {"map": [], "map_std": [], "obs": [], "eval": []}
        for n in range(0, len(bins) - 1):
            b_start = bins[n]
            b_end = bins[n + 1]
            eval_in_bin = onp.logical_and(
                log_eval_data < b_end, log_eval_data >= b_start
            )
            map_mean = onp.mean(map_data[eval_in_bin])
            map_std = onp.std(map_data[eval_in_bin])
            print(method)
            print(f"bin {n} items {np.sum(eval_in_bin)}", map_mean)
            binned_data[method]["map"].append(map_mean)
            binned_data[method]["map_std"].append(map_std)
            # binned_data[method]["obs"].append(obs_mean)
    map_vals = [binned_data[method]["map"] for method in eval_plot_data]
    map_stds = [binned_data[method]["map_std"] for method in eval_plot_data]
    # obs_vals = [binned_data[method]["obs"] for method in eval_plot_data]

    n_method = len(list(eval_plot_data.keys()))
    x = onp.arange(len(bins) - 1)  # the label locations
    width = 0.5  # the width of the bars
    for mi, method in enumerate(eval_plot_data):
        print(map_vals[mi])
        rects = ax.bar(
            x - width / n_method * mi + width / 2,
            map_vals[mi][::-1],
            width / n_method,
            yerr=map_stds[mi][::-1],
            label=method,
            color=colors[method],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels[::-1])
    # plt.xticks(x, list(data.keys()))
    plt.legend(loc="upper left")
    # ax.plot(xy_range, xy_range, "k--", linewidth=1, color="gray")

    # ax.set_ylim([-1, 2])
    # ax.set_ylim(*xy_range)

    # ax.set_xlim(1e-1, 10)
    # ax.set_ylim(1e-1, 10)
    # ax.set_xscale('log')
    ax.set_yscale("log")

    ax.set_ylabel("Posterior Regret on Proposed task")
    ax.set_xlabel("Current Posterior Regret")
    # plt.axis('scaled')
    ax.set_title(
        f"Posterior regret on next task vs Current posterior regret (higher is better)"
    )
    plt.savefig(os.path.join(exp_dir, exp_name, f"binned_map_vs_eval.png"))
    plt.show()


if __name__ == "__main__":
    N = -1
    all_seeds = [0, 1, 2, 3, 21, 22]
    not_seeds = []
    exp_dir = "data/200516"
    exp_name = (
        "active_ird_ibeta_50_joint_dbeta_20_dvar_0.1_eval_mean_128_seed_0_603_adam"
    )
    # exp_name = "active_ird_ibeta_50_w0_indep_dbeta_20_dvar_0.1_eval_mean_128_seed_0_603_adam"
    # exp_name = "active_ird_ibeta_50_w0_joint_dbeta_20_dvar_0.1_eval_mean_128_seed_0_603_adam"
    # alt_name = "active_ird_ibeta_50_true_w1_eval_unif_128_seed_0_603_adam"

    MAX_LEN = 4
    MAX_RANDOM_LEN = 4
    PADDING = 0

    use_seeds = [str(random.PRNGKey(si)) for si in all_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]
    not_method = ["ratiomin"]

    eval_plot_data = {}
    map_plot_data = {}
    obs_plot_data = {}
    load_data(eval_plot_data, map_plot_data, obs_plot_data, exp_name)
    # load_data(eval_plot_data, map_plot_data, exp_name, alt_name)
    plot_data(eval_plot_data, map_plot_data, obs_plot_data)
    # plot_binned_data(eval_plot_data, map_plot_data, obs_plot_data)
