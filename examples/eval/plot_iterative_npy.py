from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

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

colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "purple",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def plot_iterative_eval():
    all_data = []
    methods = None
    for seed in use_seeds:
        if seed in not_seeds:
            continue
        file_name = f"{exp_name}_eval_seed_{seed:02d}.npy"
        file_path = f"{exp_dir}/{exp_name}/{file_name}"
        print(file_path)
        if os.path.isfile(file_path):
            all_data.append(np.load(file_path, allow_pickle=True).item())

    # Gather active function names
    # Gather eval output for each active function
    all_eval = dict()
    all_obs = dict()
    n_xs = None
    for data in all_data:
        for fn_key in data.keys():
            if fn_key not in all_eval:
                all_eval[fn_key] = []
                all_obs[fn_key] = []
            all_eval[fn_key].append(data[fn_key]["violation"])
            all_obs[fn_key].append(data[fn_key]["obs_violation"])
            if n_xs is None:
                n_xs = len(data[fn_key]["violation"])
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.set_palette("husl")
    for i, (method, evals) in enumerate(all_eval.items()):
        sns.tsplot(
            time=range(1, 1 + n_xs),
            color=colors[method],
            data=onp.array(evals),
            condition=method,
        )
        ax.plot(
            range(1, 1 + n_xs),
            onp.array(all_obs[method]).mean(axis=0),
            color=colors[method],
            linestyle="--",
            label=method + " proxy",
        )
    plt.xticks(range(1, 1 + n_xs))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration")
    plt.ylabel("Violation")
    plt.title("Posterior Violation")
    plt.savefig(f"{exp_dir}/{exp_name}/violations.png")
    plt.show()

    # ys = [np.mean(np.array(all_eval[fn_key])) for fn_key in methods]
    # yerr = [np.std(np.array(all_eval[fn_key])) for fn_key in methods]
    # plt.bar(xs, ys, yerr=yerr)
    # plt.xticks(xs, methods)
    # plt.show()


if __name__ == "__main__":
    use_seeds = [0, 1, 2, 3]  # list(range(30))
    # use_seeds = [3]  # list(range(30))
    not_seeds = []

    not_methods = []
    exp_dir = "data/200501"
    exp_name = "iterative_divide"
    plot_iterative_eval()
