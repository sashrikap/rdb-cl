import os
import jax.numpy as np
import numpy as onp
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
from jax import random


def read_seed(path):
    # print(path)
    data = np.load(path, allow_pickle=True)["eval_hist"].item()
    return data


def cleanup(arr, max_len):
    if len(arr) == 1:
        return np.array(arr)
    else:
        # print([len(a) for a in arr])
        # print(f"{len(arr)} seeds")
        arrs = []
        for a in arr:
            # if len(a) == max_len:
            #     arrs.append(a)
            if len(a) >= max_len + PADDING:
                arrs.append(a[PADDING : PADDING + max_len])
                # arrs.append(a[:3])
        # return arrs[:2]
        # print(np.array(arrs))
        return np.array(arrs)[:, :]


def plot_perform(data_dir, data):
    sns.set_palette("husl")
    colors = "gbkr"
    for i, (method, mdict) in enumerate(data.items()):
        perf = np.array(mdict["perform"])
        sns.tsplot(
            time=range(len(perf[0])), color=colors[i], data=perf, condition=method
        )
    plt.xticks(range(len(perf[0])))
    plt.legend(loc="lower right")
    plt.xlabel("Iteration")
    plt.ylabel("Log ratio")
    plt.title("Log Relative Performance")
    plt.savefig(os.path.join(data_dir, "performance.png"))
    plt.show()
    # print(data)


def plot_data():
    seedpaths = []
    seeddata = []
    if os.path.isdir(os.path.join(exp_dir, exp_name)):
        for file in sorted(os.listdir(os.path.join(exp_dir, exp_name))):
            if exp_name in file:
                exp_path = os.path.join(exp_dir, exp_name, file)
                if os.path.isfile(exp_path):
                    use_bools = [str(s) in exp_path for s in use_seeds]
                    not_bools = [str(s) in exp_path for s in not_seeds]
                    if onp.any(use_bools) and not onp.any(not_bools):
                        seedpaths.append(exp_path)

    if os.path.isdir(os.path.join(rand_dir, rand_name)):
        for file in sorted(os.listdir(os.path.join(rand_dir, rand_name))):
            if rand_name in file:
                exp_path = os.path.join(rand_dir, rand_name, file)
                if os.path.isfile(exp_path):
                    use_bools = [str(s) in exp_path for s in use_seeds]
                    not_bools = [str(s) in exp_path for s in not_seeds]
                    if onp.any(use_bools) and not onp.any(not_bools):
                        seedpaths.append(exp_path)

    for exp_path in seedpaths:
        seeddata.append(read_seed(exp_path))
        print(exp_path, len(seeddata[-1]))

    data = {}
    for idx, (sd, spath) in enumerate(zip(seeddata, seedpaths)):
        if not all([len(h) > PADDING + MAX_LEN for h in sd.values()]):
            continue
        for method, hist in sd.items():
            if method not in data.keys():
                data[method] = {"perform": []}
            data[method]["perform"].append([])
            for h in hist:
                data[method]["perform"][-1].append(h["perform"])
            print(
                f"idx {idx} seed {spath} perf {hist[0]['perform']:.3f}, {hist[1]['perform']:.3f}"
            )
            # import pdb; pdb.set_trace()
    for method, mdict in data.items():
        if "random" in method:
            mdict["perform"] = cleanup(mdict["perform"], MAX_RANDOM_LEN)
        else:
            mdict["perform"] = cleanup(mdict["perform"], MAX_LEN)
        print(method, mdict["perform"].shape)
        # print(method, mdict["perform"])

    plot_perform(exp_dir, data)


if __name__ == "__main__":
    N = -1
    use_seeds = [1]  # list(range(20))
    # use_seeds = list(range(1))
    not_seeds = [20, 21, 22, 23, 24]
    MAX_LEN = 4
    MAX_RANDOM_LEN = 20
    PADDING = 0

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]
    # exp_dir = "data/200116"
    # exp_name = "active_ird_exp_mid"

    # exp_dir = "data/200118"
    exp_dir = "data/200120"
    exp_name = "active_ird_exp_two"
    rand_dir = "data/200116_nope"
    rand_name = "random_ird_exp_mid"
    plot_data()
