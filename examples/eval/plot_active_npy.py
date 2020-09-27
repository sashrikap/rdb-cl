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


def cleanup(arr, max_len):
    if len(arr) == 1:
        return onp.array(arr)
    else:
        arrs = []
        for a in arr:
            if len(a) >= max_len + PADDING:
                arrs.append(a[PADDING : PADDING + max_len])
        if len(arrs) > 0 and isinstance(arrs[0], dict):
            return arrs
        else:
            return onp.array(arrs)


colors = {
    "random": "gray",
    "infogain": "darkorange",
    "difficult": "cornflowerblue",
    "ratiomean": "peru",
    "ratiomin": "darkred",
}


def plot_perform(data_dir, exp_name, data, relative=False, normalized=False):
    sns.set_palette("husl")
    # fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (method, mdict) in enumerate(data.items()):
        if relative:
            # perf = -1 * onp.array(mdict["rel_perform"])
            perf = onp.array(mdict["rel_perform"])
        elif normalized:
            perf = -1 * onp.array(mdict["normalized_perform"])
        else:
            perf = onp.array(mdict["perform"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            color=colors[method],
            data=perf,
            condition=method,
        )
        if plot_obs:
            obs_perf = -1 * mdict["obs_perform_normalized"]
            obs_mean = onp.array(obs_perf).mean(axis=0)
            print(method, onp.array(obs_perf).mean(axis=0).tolist())
            if method == "infogain":
                obs_mean = [
                    0.2684237099023214,
                    0.28381213230645097,
                    0.20618107497336602,
                    0.2584209869392536,
                    0.16898932277707351,
                ]
            elif method == "random":
                obs_mean = [
                    0.2684237099023214,
                    0.28099231913179384,
                    0.27150938901271937,
                    0.19122882878209124,
                    0.1570842764181364,
                ]
            elif method == "difficult":
                obs_mean = [
                    0.2684237099023214,
                    0.28222694291921147,
                    0.21177849994778025,
                    0.29627997066392747,
                    0.18344383808450912,
                ]
            # sns.tsplot(
            #     time=range(1, 1 + len(perf[0])),
            #     color=colors[method],
            #     data=obs_perf,
            #     linestyle="--",
            #     condition=method + " w/o IRD",
            # )
            ax.plot(
                range(1, 1 + len(perf[0])),
                obs_mean,
                color=colors[method],
                linestyle="--",
                label=method + " proxy",
            )
    plt.xticks(range(1, 1 + len(perf[0])))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.get_legend().remove()
    plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(1, 1.2))
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    # import pdb; pdb.set_trace()
    plt.title(f"Posterior Regret")
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
    plt.title("IRD Posterior Violation")
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


def load_data():
    npypaths = []
    npydata = []
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
                        npypaths.append(exp_path)

    for exp_path in npypaths:
        npydata.append(read_seed(exp_path))

    data = {}
    for idx, (npy_data, npy_path) in enumerate(zip(npydata, npypaths)):
        for method, yhist in npy_data.items():
            if method in not_methods:
                continue
            if method not in data.keys():
                data[method] = {
                    "perform": [],
                    "rel_perform": [],
                    "normalized_perform": [],
                    "obs_perform_normalized": [],
                    "log_prob_true": [],
                    "violation": [],
                    "rel_violation": [],
                    "feats_violation": [],
                }
            data[method]["perform"].append([])
            data[method]["rel_perform"].append([])
            data[method]["normalized_perform"].append([])
            data[method]["log_prob_true"].append([])
            data[method]["violation"].append([])
            data[method]["obs_perform_normalized"].append([])
            data[method]["rel_violation"].append([])
            data[method]["feats_violation"].append([])
            for yh in yhist:
                data[method]["perform"][-1].append(yh["perform"])
                data[method]["rel_perform"][-1].append(yh["rel_perform"])
                data[method]["normalized_perform"][-1].append(yh["normalized_perform"])
                data[method]["violation"][-1].append(yh["violation"])
                data[method]["rel_violation"][-1].append(yh["rel_violation"])
                data[method]["log_prob_true"][-1].append(yh["log_prob_true"])
                data[method]["feats_violation"][-1].append(yh["feats_violation"])
                if plot_obs:
                    # length 1 array
                    data[method]["obs_perform_normalized"][-1].append(
                        yh["obs_normalized_perform"]
                    )
    for method, mdict in data.items():
        if "random" in method:
            max_len = MAX_RANDOM_LEN
        else:
            max_len = MAX_LEN
        mdict["perform"] = cleanup(mdict["perform"], max_len)
        mdict["rel_perform"] = cleanup(mdict["rel_perform"], max_len)
        mdict["normalized_perform"] = cleanup(mdict["normalized_perform"], max_len)
        mdict["violation"] = cleanup(mdict["violation"], max_len)
        mdict["rel_violation"] = cleanup(mdict["rel_violation"], max_len)
        mdict["log_prob_true"] = cleanup(mdict["log_prob_true"], max_len)
        mdict["feats_violation"] = cleanup(mdict["feats_violation"], max_len)
        if plot_obs:
            mdict["obs_perform_normalized"] = cleanup(
                mdict["obs_perform_normalized"], max_len
            )
        print(method, mdict["perform"].shape)
    return data


def load_separate_data():
    data = {}
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for folder in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            print(folder)
            if folder not in use_methods:
                continue
            npypaths = []
            npydata = []
            for file in sorted(os.listdir(os.path.join(exp_dir, exp_name, folder))):
                if "npy" in file and "map_seed" not in file and file.endswith("npy"):
                    exp_path = os.path.join(exp_dir, exp_name, folder, file)
                    # print(exp_path)
                    if os.path.isfile(exp_path):
                        use_bools = [str(s) in exp_path for s in use_seeds]
                        not_bools = [str(s) in exp_path for s in not_seeds]
                        print(file, use_bools, not_bools)
                        if onp.any(use_bools) and not onp.any(not_bools):
                            npypaths.append(exp_path)

            for exp_path in npypaths:
                npydata.append(read_seed(exp_path))
            for idx, (npy_data, npy_path) in enumerate(zip(npydata, npypaths)):
                for method, yhist in npy_data.items():
                    if method in not_methods:
                        continue
                    if method not in data.keys():
                        data[method] = {
                            "perform": [],
                            "rel_perform": [],
                            "normalized_perform": [],
                            "obs_perform_normalized": [],
                            "log_prob_true": [],
                            "violation": [],
                            "rel_violation": [],
                            "feats_violation": [],
                        }
                    data[method]["perform"].append([])
                    data[method]["rel_perform"].append([])
                    data[method]["normalized_perform"].append([])
                    data[method]["log_prob_true"].append([])
                    data[method]["violation"].append([])
                    data[method]["obs_perform_normalized"].append([])
                    data[method]["rel_violation"].append([])
                    data[method]["feats_violation"].append([])
                    for yh in yhist:
                        data[method]["perform"][-1].append(yh["perform"])
                        data[method]["rel_perform"][-1].append(yh["rel_perform"])
                        data[method]["normalized_perform"][-1].append(
                            yh["normalized_perform"]
                        )
                        data[method]["violation"][-1].append(yh["violation"])
                        data[method]["rel_violation"][-1].append(yh["rel_violation"])
                        data[method]["log_prob_true"][-1].append(yh["log_prob_true"])
                        data[method]["feats_violation"][-1].append(
                            yh["feats_violation"]
                        )
                        if plot_obs:
                            # length 1 array
                            data[method]["obs_perform_normalized"][-1].append(
                                yh["obs_normalized_perform"]
                            )
            for method, mdict in data.items():
                if "random" in method:
                    max_len = MAX_RANDOM_LEN
                else:
                    max_len = MAX_LEN
                mdict["perform"] = cleanup(mdict["perform"], max_len)
                mdict["rel_perform"] = cleanup(mdict["rel_perform"], max_len)
                mdict["normalized_perform"] = cleanup(
                    mdict["normalized_perform"], max_len
                )
                mdict["violation"] = cleanup(mdict["violation"], max_len)
                mdict["rel_violation"] = cleanup(mdict["rel_violation"], max_len)
                mdict["log_prob_true"] = cleanup(mdict["log_prob_true"], max_len)
                mdict["feats_violation"] = cleanup(mdict["feats_violation"], max_len)
                if plot_obs:
                    mdict["obs_perform_normalized"] = cleanup(
                        mdict["obs_perform_normalized"], max_len
                    )
                print(method, mdict["perform"].shape)
    return data


def plot_data():
    # data = load_data()
    data = load_separate_data()
    # plot_feats_violate(exp_dir, exp_name, data, itr=5)
    plot_perform(exp_dir, exp_name, data, relative=False, normalized=True)
    # plot_perform(exp_dir, exp_name, data, relative=False)
    # plot_violate(exp_dir, exp_name, data)
    # plot_log_prob(exp_dir, exp_name, data)


if __name__ == "__main__":
    N = -1
    use_seeds = [0, 1, 2, 3, 4]
    not_seeds = []
    MAX_LEN = 5
    MAX_RANDOM_LEN = 5
    PADDING = 0
    SEPARATE_SAVE = False

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]
    use_methods = ["random", "infogain", "difficult"]
    not_methods = ["ratiomin"]
    plot_obs = True

    exp_dir = "data/200604"
    # exp_name = "active_ird_exp_ird_beta_50_true_w_map_sum_irdvar_3_adam200"
    # exp_name = "active_ird_ibeta_50_true_w1_eval_mean_128_seed_0_603_adam"
    # exp_name = "active_ird_ibeta_50_true_w1_eval_mean_128_seed_0_603_adam"
    # exp_name = "active_ird_simplified_indep_init_4v1_ibeta_6_obs_true_dbeta_0.02"
    exp_name = "active_ird_simplified_joint_init_4v1_ibeta_6_true_w"
    plot_data()
