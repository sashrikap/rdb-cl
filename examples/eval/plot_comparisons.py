import matplotlib.pyplot as plt
import numpy as np
import os, yaml
from rdb.infer.dictlist import DictList


def plot_comparisons(save_dir, exp_name):
    exp_data = []
    for file in os.listdir(os.path.join(save_dir, exp_name)):
        if exp_name in file and "npy" in file:
            fname = os.path.join(save_dir, exp_name, file)
            data = np.load(fname, allow_pickle=True).item()
            exp_data.append(data)

            if "[0 1]" in file:
                print("designer", np.array(data["designer"]).mean())
                print("joint", np.array(data["joint"]).mean())
                print("divide", np.array(data["divide"]).mean())

    designer_data = np.array([d["designer"] for d in exp_data])
    designer_info = DictList([d["designer_feats"] for d in exp_data])
    divide_data = np.array([d["divide"] for d in exp_data])
    divide_info = DictList([d["divide_feats"] for d in exp_data])
    joint_data = np.array([d["joint"] for d in exp_data])
    joint_info = DictList([d["joint_feats"] for d in exp_data])

    y = [np.mean(joint_data), np.mean(divide_data), np.mean(designer_data)]
    yerr = [np.std(joint_data), np.std(divide_data), np.std(designer_data)]

    fig, ax = plt.subplots()
    plt.bar(np.arange(3), y, yerr=yerr)
    plt.xticks(np.arange(3), ["Joint IRD", "Divide&Conquer", "Joint Designer"])
    plt.title("Violations")
    figpath = f"{save_dir}/{exp_name}/compare.png"
    plt.savefig(figpath)

    fig, ax = plt.subplots()
    ys = np.zeros(3)
    for key in designer_info[0].keys():
        joint_info_key = joint_info[key].mean()
        divide_info_key = divide_info[key].mean()
        designer_info_key = designer_info[key].mean()
        y_key = [joint_info_key, divide_info_key, designer_info_key]
        print(key, y_key)
        # import pdb; pdb.set_trace()
        plt.bar(np.arange(3), y_key, bottom=ys, label=key)
        ys += np.array(y_key)
    # import pdb; pdb.set_trace()
    plt.xticks(np.arange(3), ["Joint IRD", "Divide&Conquer", "Joint Designer"])
    plt.title("Violations")
    plt.legend()
    figpath = f"{save_dir}/{exp_name}/compare_features.png"
    plt.savefig(figpath)


if __name__ == "__main__":
    save_dir = "data/200229"
    exp_name = "compare_sum_beta_20_true_w_irdvar_3_602_adam_design_04_num_05"
    plot_comparisons(save_dir, exp_name)
