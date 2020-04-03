from rdb.infer import *
import numpy as np


def load_file(save_path):
    data = np.load(save_path, allow_pickle=True)
    # ps.log_prob(weights[0])
    tasks = data["curr_tasks"].item()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    save_path = "data/200401/active_ird_ibeta_50_true_w1_eval_unif_128_seed_0_603_adam/active_ird_ibeta_50_true_w1_eval_unif_128_seed_0_603_adam_seed_[ 0 21].npz"
    load_file(save_path)
