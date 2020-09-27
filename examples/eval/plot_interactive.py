import os
import jax.numpy as jnp
import numpy as onp
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
from jax import random


def read_seed(path):
    # print(path)
    data = jnp.load(path, allow_pickle=True)["eval_hist"].item()
    # keys = list(data.keys())
    # assert len(keys) == 1
    # values = [d["violation"] for d in data[keys[0]]]
    # return values
    return data


def cleanup(arr, max_len):
    if len(arr) == 1:
        return onp.array(arr)
    else:
        arrs = []
        for a in arr:
            if len(a) >= max_len + PADDING:
                arrs.append(a)
        try:
            return onp.array(arrs)[:, :]
        except:
            import pdb

            pdb.set_trace()


def plot_vio(data_dir, data):
    sns.set_palette("husl")
    colors = "gbkr"
    for i, (method, mdict) in enumerate(data.items()):
        vio = jnp.array(mdict["violation"])
        sns.tsplot(time=range(len(vio[0])), color=colors[i], data=vio, condition=method)
    # import
    # plt.xticks(range(len(vio[0])))
    plt.legend(loc="upper right")
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("violation")
    plt.savefig(os.path.join(data_dir, "violation.png"))
    plt.show()
    # print(data)


def plot_data():
    seedpaths = []
    seeddata = []
    # assert os.path.isdir(os.path.join(exp_dir, exp_name))
    for dir_name in sorted(os.listdir(exp_dir)):
        if exp_name not in dir_name:
            continue
        # data/200120_eval/interactive_ird_exp_three_random
        exp_path = os.path.join(exp_dir, dir_name)
        for file in os.listdir(exp_path):
            file_path = os.path.join(exp_path, file)
            if os.path.isfile(file_path):
                use_bools = [str(s) in file_path for s in use_seeds]
                not_bools = [str(s) in file_path for s in not_seeds]
                if onp.any(use_bools) and not onp.any(not_bools):
                    seedpaths.append(file_path)

    for file_path in seedpaths:
        seeddata.append(read_seed(file_path))
        print(file_path, len(seeddata[-1]))

    data = {}
    for idx, (sd, spath) in enumerate(zip(seeddata, seedpaths)):
        if not all([len(h) >= PADDING + MAX_LEN for h in sd.values()]):
            continue
        for method, hist in sd.items():
            if method not in use_methods:
                continue
            if method not in data.keys():
                data[method] = {"violation": []}
            data[method]["violation"].append([])
            for h in hist:
                data[method]["violation"][-1].append(h["violation"])
            print(
                f'idx {idx} seed {spath} violation {hist[0]["violation"]:.3f}, {hist[1]["violation"]:.3f}'
            )
            # import pdb; pdb.set_trace()
    for method, mdict in data.items():
        mdict["violation"] = cleanup(mdict["violation"], MAX_LEN)
        # print(method, mdict["violation"].shape)
        print(method, mdict["violation"])

    plot_vio(exp_dir, data)


if __name__ == "__main__":
    N = -1
    # use_seeds = list(range(20))
    use_seeds = [0]
    not_seeds = []
    use_methods = ["infogain", "ratiomean", "ratiomin", "random"]
    # use_methods = ["random"]
    MAX_LEN = 0
    MAX_RANDOM_LEN = 20
    PADDING = 0

    use_seeds = [str(random.PRNGKey(si)) for si in use_seeds]
    not_seeds = [str(random.PRNGKey(si)) for si in not_seeds]

    exp_dir = "data/test"
    exp_name = "test_interactive_ird_exp_natural_one"
    plot_data()
