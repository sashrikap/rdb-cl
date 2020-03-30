from rdb.infer import *
import numpy as np


def load_file(save_path):
    data = np.load(save_path, allow_pickle=True)
    # ps.log_prob(weights[0])
    tasks = data["curr_tasks"].item()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    save_path = "data/200320/active_ird_sum_ibeta_10_irdvar_3_true_w_w1_eval_unif_602_adam/active_ird_sum_ibeta_10_irdvar_3_true_w_w1_eval_unif_602_adam_seed_[0 1].npz"
    load_file(save_path)
