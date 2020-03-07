import matplotlib.pyplot as plt
import numpy as np
import os, yaml
from rdb.infer.dictlist import DictList


def plot_comparisons(save_dir, exp_name):
    exp_data = []
    for file in os.listdir(os.path.join(save_dir, exp_name)):
        if exp_name in file and "npy" in file and "map" not in file:
            print(file)
            fname = os.path.join(save_dir, exp_name, file)
            data = np.load(fname, allow_pickle=True).item()
            exp_data.append(data)

            print("Joint proposal")
            for method in data["joint_proposal"].keys():
                print("Method", method)
                max_ids = np.argsort(data["joint_proposal"][method]["scores"])[-3:]
                print(
                    "Task",
                    max_ids[-1],
                    data["joint_proposal"][method]["candidates"][max_ids[-1]],
                )
                print(
                    "Task",
                    max_ids[-2],
                    data["joint_proposal"][method]["candidates"][max_ids[-2]],
                )

            print("Divide proposal")
            for method in data["divide_proposal"].keys():
                print("Method", method)
                max_id = data["divide_proposal"][method]["scores"].argmax()
                print(
                    "Task",
                    max_id,
                    data["divide_proposal"][method]["candidates"][max_id],
                )
            print()


if __name__ == "__main__":
    save_dir = "data/200229"
    exp_name = "compare_sum_beta_20_true_w_irdvar_3_602_adam_design_02_num_06"
    plot_comparisons(save_dir, exp_name)
