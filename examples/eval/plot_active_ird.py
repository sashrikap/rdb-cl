import os
import jax.numpy as np
import numpy as onp
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt


def read_seed(path):
    # print(path)
    data = np.load(path, allow_pickle=True)["eval_hist"].item()
    return data


def cleanup(arr):
    if len(arr) == 1:
        return np.array(arr)
    else:
        max_len = max([len(a) for a in arr])
        # print([len(a) for a in arr])
        # print(f"{len(arr)} seeds")
        arrs = []
        for a in arr:
            # if len(a) == max_len:
            #     arrs.append(a)
            if len(a) > MAX_LEN:
                arrs.append(a[:MAX_LEN])
                # arrs.append(a[:3])
        # return arrs[:2]
        # print(np.array(arrs))
        return np.array(arrs)[:N, :]


def plot_perform(path, data):
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
    plt.savefig(os.path.join(path, "performance.png"))
    plt.show()
    # print(data)


def plot_data(data_path):
    seedpaths = []
    seeddata = []
    dir_path = os.path.dirname(data_path)
    filename = os.path.basename(data_path)
    for file in sorted(os.listdir(dir_path)):
        if filename in file:
            filepath = os.path.join(dir_path, file)
            if os.path.isfile(filepath):
                if seed in filepath:
                    seedpaths.append(filepath)

    for path in seedpaths:
        seeddata.append(read_seed(path))

    data = {}
    for idx, sd in enumerate(seeddata):
        if not all([len(h) > MAX_LEN for h in sd.values()]):
            continue
        for method, hist in sd.items():
            if method not in data.keys():
                data[method] = {"perform": []}
            data[method]["perform"].append([])
            for h in hist:
                data[method]["perform"][-1].append(h["perform"])
            # import pdb; pdb.set_trace()
    for method, mdict in data.items():
        mdict["perform"] = cleanup(mdict["perform"])
        print(method, mdict["perform"].shape)

    plot_perform(data_path, data)


if __name__ == "__main__":
    N = -1
    # seed = "[ 0 13]"
    seed = ""
    MAX_LEN = 8
    # MAX_LEN = 4
    # data_path = "data/191216/active_ird_exp1"
    # data_path = "data/191217/active_ird_exp1"
    # data_path = "data/200103/active_ird_exp_mid"
    data_path = "data/200110/active_ird_exp_mid"
    plot_data(data_path)
