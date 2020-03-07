from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


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


def plot_data():
    seedpaths = []
    seeddata = []
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for file in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            if (
                exp_name in file
                and "npy" in file
                and "map" in file
                and file.endswith("npy")
            ):
                print(file)
                exp_path = os.path.join(exp_dir, exp_name, file)
                if os.path.isfile(exp_path):
                    use_bools = [str(s) in exp_path for s in use_seeds]
                    not_bools = [str(s) in exp_path for s in not_seeds]
                    if onp.any(use_bools) and not onp.any(not_bools):
                        seedpaths.append(exp_path)

    exp_path = seedpaths[0]
    exp_read = read_seed(exp_path)

    data = {}
    fig, ax = plt.subplots()
    idx = 0
    for method, hist in exp_read.items():
        data[method] = {
            "perform": np.array(hist[idx]["map_perform"])
            - np.array(hist[idx]["obs_perform"]),
            "violation": np.array(hist[idx]["map_violation"])
            - np.array(hist[idx]["obs_violation"]),
        }
        # Only look at first proposed task

    x = np.arange(len(list(data.keys())))
    dim = 2
    w = 0.75
    dimw = w / dim
    ax.bar(
        x,
        [data[m]["perform"].mean() for m in data.keys()],
        dimw,
        yerr=[data[m]["perform"].std() for m in data.keys()],
        label="Performance",
    )
    ax.bar(
        x + dimw,
        [data[m]["violation"].mean() for m in data.keys()],
        dimw,
        yerr=[data[m]["violation"].std() for m in data.keys()],
        label="Violations",
    )

    colors = "gbkr"
    plt.xticks(x, list(data.keys()))
    plt.legend(loc="upper right")
    plt.xlabel("Method")
    plt.ylabel("MAP - Joint")
    plt.title(f"MAP performance on next proposed task (seed {use_seeds[0]})")
    plt.savefig(os.path.join(exp_dir, exp_name, f"map_vs_obs_seed_{use_seeds[0]}.png"))


if __name__ == "__main__":
    N = -1
    all_seeds = [0, 1, 2, 3, 4, 5]
    exp_dir = "data/200229"
    # exp_name = "active_ird_exp_ird_beta_50_true_w_map_sum_irdvar_3_adam200"
    exp_name = "active_ird_sum_beta_10_dprior_2_irdvar_3_dvar_1_602_adam"

    for seed in all_seeds:
        use_seeds = [seed]
        not_seeds = []
        MAX_LEN = 8
        MAX_RANDOM_LEN = 8
        PADDING = 0

        use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
        not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]

        plot_data()
