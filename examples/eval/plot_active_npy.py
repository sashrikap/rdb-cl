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
        return onp.array(arr)
    else:
        arrs = []
        for a in arr:
            if len(a) >= max_len + PADDING:
                arrs.append(a[PADDING : PADDING + max_len])
        if isinstance(arrs[0], dict):
            return arrs
        else:
            return onp.array(arrs)


colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "purple",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def plot_perform(data_dir, exp_name, data, relative=False):
    sns.set_palette("husl")
    for i, (method, mdict) in enumerate(data.items()):
        if relative:
            # perf = -1 * onp.array(mdict["rel_perform"])
            perf = onp.array(mdict["rel_perform"])
        else:
            perf = onp.array(mdict["perform"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            color=colors[method],
            data=perf,
            condition=method,
        )
    plt.xticks(range(1, 1 + len(perf[0])))
    plt.legend(loc="lower right")
    plt.xlabel("Iteration")
    plt.ylabel("Performance Gap")
    # import pdb; pdb.set_trace()
    plt.title(f"{'Relative ' if relative else ''}IRD Posterior vs True_w Performance")
    plt.savefig(os.path.join(data_dir, exp_name, "performance.png"))
    plt.tight_layout()
    plt.show()


def plot_violate(data_dir, exp_name, data):
    sns.set_palette("husl")
    for i, (method, mdict) in enumerate(data.items()):
        perf = onp.array(mdict["violation"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            color=colors[method],
            data=perf,
            condition=method,
        )
    plt.xticks(range(1, 1 + len(perf[0])))
    plt.legend(loc="lower right")
    plt.xlabel("Iteration")
    plt.ylabel("Violation Gap")
    plt.title("IRD Posterior vs True_w Violation")
    plt.savefig(os.path.join(data_dir, exp_name, "violation.png"))
    plt.tight_layout()
    plt.show()


def plot_log_prob(data_dir, exp_name, data):
    sns.set_palette("husl")
    for i, (method, mdict) in enumerate(data.items()):
        perf = onp.array(mdict["log_prob_true"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            color=colors[method],
            data=perf,
            condition=method,
        )
    plt.xticks(range(1, 1 + len(perf[0])))
    plt.legend(loc="lower right")
    plt.xlabel("Iteration")
    plt.ylabel("Posterior - True W")
    plt.title("Log Prob Of true reward")
    plt.savefig(os.path.join(data_dir, exp_name, "logprobtrue.png"))
    plt.tight_layout()
    plt.show()


def plot_feats_violate(data_dir, exp_name, data, itr=-1):
    sns.set_palette("husl")
    all_vios = {}
    all_vios_sum = {}
    for i, (method, mdict) in enumerate(data.items()):
        # nkeys * niters
        all_vios[method] = {}
        for rng_feats_vios in mdict["feats_violation"]:
            # niters
            feats_vios = rng_feats_vios[itr]
            for feats_key, n_vios in feats_vios.items():
                if feats_key not in all_vios[method]:
                    all_vios[method][feats_key] = [n_vios.mean()]
                else:
                    all_vios[method][feats_key].append(n_vios.mean())
        for feats_key in all_vios[method].keys():
            all_vios[method][feats_key] = onp.mean(all_vios[method][feats_key])
        all_vios_sum[method] = onp.sum(list(all_vios[method].values()))
        print(method, all_vios[method])
    n_methods = len(list(data.keys()))
    barlist = plt.bar(onp.arange(n_methods), onp.array(list(all_vios_sum.values())))
    for i in range(n_methods):
        barlist[i].set_color(colors[list(data.keys())[i]])
    plt.xticks(onp.arange(n_methods), all_vios_sum.keys())
    plt.legend(loc="upper right")
    plt.xlabel("Method")
    plt.ylabel("MAP - True W")
    plt.title("Violation by features")
    plt.savefig(os.path.join(data_dir, exp_name, "violation_features.png"))
    plt.tight_layout()
    plt.show()


def plot_data():
    seedpaths = []
    seeddata = []
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for file in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            if (
                # exp_name in file
                # and "npy" in file
                "npy" in file
                and "map_seed" not in file
                and file.endswith("npy")
            ):
                exp_path = os.path.join(exp_dir, exp_name, file)
                # print(exp_path)
                if os.path.isfile(exp_path):
                    use_bools = [str(s) in exp_path for s in use_seeds]
                    not_bools = [str(s) in exp_path for s in not_seeds]
                    print(file, use_bools, not_bools)
                    if onp.any(use_bools) and not onp.any(not_bools):
                        seedpaths.append(exp_path)

    for exp_path in seedpaths:
        exp_read = read_seed(exp_path)
        # print(exp_path)
        # print(exp_path, len(exp_read["infogain"]))
        seeddata.append(exp_read)

    data = {}
    for idx, (sd, spath) in enumerate(zip(seeddata, seedpaths)):
        for method, hist in sd.items():
            if method in not_methods:
                continue
            if method not in data.keys():
                data[method] = {
                    "perform": [],
                    "rel_perform": [],
                    "log_prob_true": [],
                    "violation": [],
                    "rel_violation": [],
                    "feats_violation": [],
                }
            data[method]["perform"].append([])
            data[method]["rel_perform"].append([])
            data[method]["log_prob_true"].append([])
            data[method]["violation"].append([])
            data[method]["rel_violation"].append([])
            data[method]["feats_violation"].append([])
            for h in hist:
                data[method]["perform"][-1].append(h["perform"])
                data[method]["rel_perform"][-1].append(h["rel_perform"])
                data[method]["violation"][-1].append(h["violation"])
                data[method]["rel_violation"][-1].append(h["rel_violation"])
                data[method]["log_prob_true"][-1].append(h["log_prob_true"])
                data[method]["feats_violation"][-1].append(h["feats_violation"])
    for method, mdict in data.items():
        if "random" in method:
            mdict["perform"] = cleanup(mdict["perform"], MAX_RANDOM_LEN)
            mdict["rel_perform"] = cleanup(mdict["rel_perform"], MAX_RANDOM_LEN)
            mdict["violation"] = cleanup(mdict["violation"], MAX_RANDOM_LEN)
            mdict["rel_violation"] = cleanup(mdict["rel_violation"], MAX_RANDOM_LEN)
            mdict["log_prob_true"] = cleanup(mdict["log_prob_true"], MAX_RANDOM_LEN)
            mdict["feats_violation"] = cleanup(mdict["feats_violation"], MAX_RANDOM_LEN)
        else:
            mdict["perform"] = cleanup(mdict["perform"], MAX_LEN)
            mdict["rel_perform"] = cleanup(mdict["rel_perform"], MAX_LEN)
            mdict["violation"] = cleanup(mdict["violation"], MAX_LEN)
            mdict["rel_violation"] = cleanup(mdict["rel_violation"], MAX_LEN)
            mdict["log_prob_true"] = cleanup(mdict["log_prob_true"], MAX_LEN)
            mdict["feats_violation"] = cleanup(mdict["feats_violation"], MAX_LEN)
        print(method, mdict["perform"].shape)

    # plot_feats_violate(exp_dir, exp_name, data, itr=5)
    plot_perform(exp_dir, exp_name, data, relative=True)
    # plot_perform(exp_dir, exp_name, data, relative=False)
    # plot_violate(exp_dir, exp_name, data)
    # plot_log_prob(exp_dir, exp_name, data)


if __name__ == "__main__":
    N = -1
    use_seeds = list(range(30))
    not_seeds = []
    MAX_LEN = 9
    MAX_RANDOM_LEN = 9
    PADDING = 0

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]
    not_methods = []

    exp_dir = "data/200521"
    # exp_name = "active_ird_exp_ird_beta_50_true_w_map_sum_irdvar_3_adam200"
    # exp_name = "active_ird_ibeta_50_true_w1_eval_mean_128_seed_0_603_adam"
    # exp_name = "active_ird_ibeta_50_true_w1_eval_mean_128_seed_0_603_adam"
    exp_name = (
        "active_ird_ibeta_50_w1_joint_dbeta_20_dvar_0.1_eval_mean_128_seed_0_603_adam"
    )
    plot_data()
