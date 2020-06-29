from jax import random
import os
import yaml
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import copy

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
        data = np.load(path, allow_pickle=True).item()
    elif path.endswith("npz"):
        data = np.load(path, allow_pickle=True)
    return data


colors = {"ird": "darkorange", "designer": "cornflowerblue"}


def plot_perform(
    data_dir, exp_name, designer_data, ird_data, relative=False, normalized=False
):
    sns.set_palette("husl")
    # fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    for data, label in zip([designer_data, ird_data], ["designer", "ird"]):
        if relative:
            perf = onp.array(designer_data["rel_perform"])
        elif normalized:
            perf = -1 * onp.array(data["normalized_perform"])
        else:
            perf = onp.array(data["perform"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            data=perf,
            color=colors[label],
            condition=label,
        )

    plt.xticks(range(1, 1 + len(perf[0])))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.legend(loc="upper center", ncol=3, bbox_to_anchor=(1, 1.2))
    plt.xlabel("Betas")
    plt.ylabel("Regret")
    # import pdb; pdb.set_trace()
    plt.title(f"Posterior Regret")
    plt.savefig(os.path.join(data_dir, exp_name, "performance.png"))
    plt.tight_layout()
    plt.show()


def plot_log_prob(data_dir, exp_name, data):
    sns.set_palette("husl")
    for i, (beta, mdict) in enumerate(data.items()):
        perf = onp.array(mdict["log_prob_true"])
        sns.tsplot(
            time=range(1, 1 + len(perf[0])),
            # color=colors[beta],
            data=perf,
            condition=beta,
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
    for i, (beta, mdict) in enumerate(data.items()):
        # nkeys * niters
        all_vios[beta] = {}
        for rng_feats_vios in mdict["feats_violation"]:
            # niters
            feats_vios = rng_feats_vios[itr]
            for feats_key, n_vios in feats_vios.items():
                if feats_key not in all_vios[beta]:
                    all_vios[beta][feats_key] = [n_vios.mean()]
                else:
                    all_vios[beta][feats_key].append(n_vios.mean())
        for feats_key in all_vios[beta].keys():
            all_vios[beta][feats_key] = onp.mean(all_vios[beta][feats_key])
        all_vios_sum[beta] = onp.sum(list(all_vios[beta].values()))
        print(beta, all_vios[beta])
    n_betas = len(list(data.keys()))
    barlist = plt.bar(onp.arange(n_betas), onp.array(list(all_vios_sum.values())))
    for i in range(n_betas):
        barlist[i].set_color(colors[list(data.keys())[i]])
    plt.xticks(onp.arange(n_betas), all_vios_sum.keys())
    plt.legend(loc="upper right")
    plt.xlabel("Beta")
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
            if "npy" in file and file.endswith("npy"):
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

    data = dict(
        perform=[],
        rel_perform=[],
        normalized_perform=[],
        violation=[],
        rel_violation=[],
        feats_violation=[],
    )
    designer_data = copy.deepcopy(data)
    ird_data = copy.deepcopy(data)
    for idx, (npy_data, npy_path) in enumerate(zip(npydata, npypaths)):
        if "designer" in npy_path:
            data = designer_data
        else:
            assert "ird" in npy_path
            data = ird_data
        data["perform"].append([])
        data["rel_perform"].append([])
        data["normalized_perform"].append([])
        data["violation"].append([])
        data["rel_violation"].append([])
        data["feats_violation"].append([])

        for beta, yhist in npy_data.items():
            data["perform"][-1].append(yhist["perform"])
            data["rel_perform"][-1].append(yhist["rel_perform"])
            data["normalized_perform"][-1].append(yhist["normalized_perform"])
            data["violation"][-1].append(yhist["violation"])
            data["rel_violation"][-1].append(yhist["rel_violation"])
            data["feats_violation"][-1].append(yhist["feats_violation"])

    return designer_data, ird_data


# def load_separate_data():
#     data = {}
#     if os.path.isdir(os.path.join(exp_dir, exp_name)):
#         for folder in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
#             print(folder)
#             if folder not in use_methods:
#                 continue
#             npypaths = []
#             npydata = []
#             for file in sorted(os.listdir(os.path.join(exp_dir, exp_name, folder))):
#                 if "npy" in file and "map_seed" not in file and file.endswith("npy"):
#                     exp_path = os.path.join(exp_dir, exp_name, folder, file)
#                     # print(exp_path)
#                     if os.path.isfile(exp_path):
#                         use_bools = [str(s) in exp_path for s in use_seeds]
#                         not_bools = [str(s) in exp_path for s in not_seeds]
#                         print(file, use_bools, not_bools)
#                         if onp.any(use_bools) and not onp.any(not_bools):
#                             npypaths.append(exp_path)

#             for exp_path in npypaths:
#                 npydata.append(read_seed(exp_path))
#             for idx, (npy_data, npy_path) in enumerate(zip(npydata, npypaths)):
#                 for method, yhist in npy_data.items():
#                     if method in not_methods:
#                         continue
#                     if method not in data.keys():
#                         data[method] = {
#                             "perform": [],
#                             "rel_perform": [],
#                             "normalized_perform": [],
#                             "obs_perform_normalized": [],
#                             "log_prob_true": [],
#                             "violation": [],
#                             "rel_violation": [],
#                             "feats_violation": [],
#                         }
#                     data[method]["perform"].append([])
#                     data[method]["rel_perform"].append([])
#                     data[method]["normalized_perform"].append([])
#                     data[method]["log_prob_true"].append([])
#                     data[method]["violation"].append([])
#                     data[method]["obs_perform_normalized"].append([])
#                     data[method]["rel_violation"].append([])
#                     data[method]["feats_violation"].append([])
#                     for yh in yhist:
#                         data[method]["perform"][-1].append(yh["perform"])
#                         data[method]["rel_perform"][-1].append(yh["rel_perform"])
#                         data[method]["normalized_perform"][-1].append(
#                             yh["normalized_perform"]
#                         )
#                         data[method]["violation"][-1].append(yh["violation"])
#                         data[method]["rel_violation"][-1].append(yh["rel_violation"])
#                         data[method]["log_prob_true"][-1].append(yh["log_prob_true"])
#                         data[method]["feats_violation"][-1].append(
#                             yh["feats_violation"]
#                         )
#                         if plot_obs:
#                             # length 1 array
#                             data[method]["obs_perform_normalized"][-1].append(
#                                 yh["obs_normalized_perform"]
#                             )
#             for method, mdict in data.items():
#                 if "random" in method:
#                     max_len = MAX_RANDOM_LEN
#                 else:
#                     max_len = MAX_LEN
#                 mdict["perform"] = cleanup(mdict["perform"], max_len)
#                 mdict["rel_perform"] = cleanup(mdict["rel_perform"], max_len)
#                 mdict["normalized_perform"] = cleanup(
#                     mdict["normalized_perform"], max_len
#                 )
#                 mdict["violation"] = cleanup(mdict["violation"], max_len)
#                 mdict["rel_violation"] = cleanup(mdict["rel_violation"], max_len)
#                 mdict["log_prob_true"] = cleanup(mdict["log_prob_true"], max_len)
#                 mdict["feats_violation"] = cleanup(mdict["feats_violation"], max_len)
#                 if plot_obs:
#                     mdict["obs_perform_normalized"] = cleanup(
#                         mdict["obs_perform_normalized"], max_len
#                     )
#                 print(method, mdict["perform"].shape)
#     return data


def plot_data():
    designer_data, ird_data = load_data()
    # data = load_separate_data()
    plot_perform(
        exp_dir, exp_name, designer_data, ird_data, relative=False, normalized=True
    )


if __name__ == "__main__":
    N = -1
    use_seeds = [0, 1, 2, 3, 4]
    not_seeds = []
    PADDING = 0
    SEPARATE_SAVE = False

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]
    plot_obs = True

    exp_dir = "data/200621"
    exp_name = "task_beta_simplified_joint_init4v1"
    plot_data()
