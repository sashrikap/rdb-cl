from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_proposal_eval():
    all_data = []
    methods = None
    for seed in use_seeds:
        if seed in not_seeds:
            continue
        file_name = f"proposal_rng_{seed:02d}.npy"
        file_path = f"{exp_dir}/{exp_name}/{file_name}"
        if os.path.isfile(file_path):
            all_data.append(np.load(file_path, allow_pickle=True).item())

    # Gather active function names
    for data in all_data:
        prop_data = data["proposal_eval"]
        if methods is None:
            methods = list(prop_data.keys())
        else:
            assert set(methods) == set(list(prop_data.keys()))
    # Gather eval output for each active function
    all_eval = {fn_key: [] for fn_key in methods}
    for data in all_data:
        for fn_key in methods:
            all_eval[fn_key].extend(
                [d["violation"] for d in data["proposal_eval"][fn_key]]
            )
    fig, ax = plt.subplots()
    xs = np.arange(len(methods))
    ys = [np.mean(np.array(all_eval[fn_key])) for fn_key in methods]
    yerr = [np.std(np.array(all_eval[fn_key])) for fn_key in methods]
    plt.bar(xs, ys, yerr=yerr)
    plt.xticks(xs, methods)
    plt.show()


def plot_training_proposal_eval(mode="bar"):
    assert mode in {"bar", "dot"}

    all_proposal_data = []
    all_training_data = []
    methods = None
    for seed in use_seeds:
        if seed in not_seeds:
            continue
        proposal_file_name = f"proposal_rng_{seed:02d}.npy"
        proposal_file_path = f"{exp_dir}/{exp_name}/{proposal_file_name}"
        if os.path.isfile(proposal_file_path):
            all_proposal_data.append(
                np.load(proposal_file_path, allow_pickle=True).item()
            )
        training_file_name = f"training_rng_{seed:02d}.npy"
        training_file_path = f"{exp_dir}/{exp_name}/{training_file_name}"
        if os.path.isfile(training_file_path):
            all_training_data.append(
                np.load(training_file_path, allow_pickle=True).item()
            )

    # Gather active function names
    for data in all_proposal_data:
        prop_data = data["proposal_eval"]
        if methods is None:
            methods = list(prop_data.keys())
        else:
            assert set(methods) == set(list(prop_data.keys()))
    # Gather eval output for each active function
    all_proposal_eval = {fn_key: [] for fn_key in methods}
    all_training_eval = {fn_key: [] for fn_key in methods}
    for data in all_proposal_data:
        for fn_key in methods:
            all_proposal_eval[fn_key].append(
                [d["violation"] for d in data["proposal_eval"][fn_key]]
            )
    for data in all_training_data:
        for fn_key in methods:
            all_training_eval[fn_key].append(data["training_eval"][fn_key]["violation"])

    colors = {
        "random": "gray",
        "infogain": "darkorange",
        "difficult": "purple",
        "ratiomean": "peru",
        "ratiomin": "darkred",
    }

    if mode == "dot":
        fig, ax = plt.subplots(figsize=(8, 8))
        for fn_key in methods:
            xs = [d for d in all_training_eval[fn_key]]
            ys = [np.mean(np.array(d)) for d in all_proposal_eval[fn_key]]
            # print(fn_key, xs)
            # xs, ys = [], []
            # for rand_idx, d in enumerate(all_proposal_eval[fn_key]):
            #     ys.extend(d)
            #     xs.extend([all_training_eval[fn_key][rand_idx]] * len(d))
            # plt.plot(xs, onp.array(ys), marker='o', linestyle='None', label=fn_key)
            plt.plot(
                xs,
                onp.array(xs) - onp.array(ys),
                marker="o",
                linestyle="None",
                label=fn_key,
                color=colors[fn_key],
                markersize=9,
            )
        plt.title("MAP Improvement")
        plt.xlabel("Training Violations")
        plt.ylabel("Improvement")
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(exp_dir, exp_name, f"proposal_{mode}.png"))
        # plt.show()
    elif mode == "bar":
        fig, ax = plt.subplots(figsize=(12, 8))
        bars_before = {fn_key: [] for fn_key in methods}
        bars_after = {fn_key: [] for fn_key in methods}
        bars_after_err = {fn_key: [] for fn_key in methods}
        width = 0.35
        for fi, fn_key in enumerate(methods):
            bars_before[fn_key] = [d for d in all_training_eval[fn_key]]
            bars_after[fn_key] = [
                np.mean(np.array(d)) for d in all_proposal_eval[fn_key]
            ]
            bars_after_err[fn_key] = [
                np.std(np.array(d)) for d in all_proposal_eval[fn_key]
            ]
            offset = -width * (float(len(methods)) - 1) / 2 + fi * width
            x = onp.arange(len(bars_before[fn_key])) * width * 4
            y = onp.array(bars_before[fn_key]) - onp.array(bars_after[fn_key])
            rects = ax.bar(x + offset, y, width, label=fn_key, color=colors[fn_key])
        ax.set_xticks(x)
        ax.set_xticklabels([f"Sample {i}" for i in range(len(use_seeds))])
        plt.xlabel("Training Tasks")
        plt.title("MAP Improvement")
        plt.ylabel("Violation Improvement")
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(os.path.join(exp_dir, exp_name, f"proposal_{mode}.png"))


if __name__ == "__main__":
    use_seeds = [0, 1, 2, 3]  # list(range(30))
    not_seeds = []

    not_methods = []
    exp_dir = "data/200413"
    # exp_name = "active_ird_exp_ird_beta_50_true_w_map_sum_irdvar_3_adam200"
    exp_name = "interactive_proposal_divide_training_03_propose_04"
    # plot_proposal_eval()
    plot_training_proposal_eval("bar")
    # plot_training_proposal_eval("dot")
