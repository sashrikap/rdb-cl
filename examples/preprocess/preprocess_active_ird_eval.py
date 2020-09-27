"""Temporary script used to fill in some missing fields of saved
experiment data. Easily outdated
"""
import os
import jax.numpy as jnp
import gym, rdb.envs.drive2d

from jax import random
from os.path import join
from tqdm import tqdm
from rdb.exps.utils import str_to_key
from rdb.infer.utils import random_choice
from numpyro.handlers import scale, condition, seed


def load_data(env, exp_dir, exp_name):
    global TEST_DATA
    all_data = {}
    for file in tqdm(os.listdir(exp_dir)):
        # Load eval data
        file_path = join(exp_dir, file)

        if not ".npz" in file_path:
            continue

        file_seed = file_path.replace(".npz", "")[
            file_path.index("seed_") + len("seed_") :
        ]
        rng_key = str_to_key(file_seed)

        eval_data = jnp.load(file_path, allow_pickle=True)
        eval_hist = eval_data["eval_hist"].item()
        curr_obs = eval_data["curr_obs"].item()
        curr_tasks = eval_data["curr_tasks"].item()
        try:
            candidate_tasks = list(eval_data["candidate_tasks"])
        except:
            candidate_tasks = None

        if VERBOSE:
            for fn_key, tasks in curr_tasks.items():
                print(f"{fn_key} tasks {tasks}")

        eval_data = dict(
            eval_hist=eval_hist,
            curr_obs=curr_obs,
            curr_tasks=curr_tasks,
            candidate_tasks=candidate_tasks,
        )

        # Load sample data
        weight_samples = {}
        weight_dir = join(exp_dir, exp_name, "save")
        for fn_key in curr_tasks.keys():
            weight_files = sorted(
                [f for f in os.listdir(weight_dir) if fn_key in f and str(rng_key) in f]
            )

            weight_samples[fn_key] = []
            for file in weight_files:
                weight_filepath = join(weight_dir, file)
                weights = list(np.load(weight_filepath, allow_pickle=True)["weights"])
                weight_samples[fn_key].append(weights)

        num_iters = max([len(tasks) for tasks in curr_tasks.values()])
        # Add in missing fields
        if TEST_DATA is not None:
            CAND_TASKS = TEST_DATA[str(rng_key)][1]["candidate_tasks"]
            for itr in range(1, num_iters):
                for fn_key in FN_KEYS:
                    if len(curr_tasks[fn_key]) > itr:
                        if len(CAND_TASKS) < itr:
                            print(f"short key {str(rng_key)} fn {fn_key} iter {itr}")
                        else:
                            candidates = CAND_TASKS[itr - 1]
                            equals = [
                                jnp.allclose(c, curr_tasks[fn_key][itr])
                                for c in candidates
                            ]
                            if not sum(equals) > 0:
                                print(f"bad key {str(rng_key)} fn {fn_key} iter {itr}")
                                assert False
            eval_data["candidate_tasks"] = CAND_TASKS
        # More missing fields
        eval_data["seed"] = str(rng_key)

        all_data[str(rng_key)] = (rng_key, eval_data, weight_samples)
    return all_data


def save_data(env, all_data, exp_dir, exp_name):
    for key_str, (rng_key, eval_data, weight_samples) in all_data.items():
        # Save eval data
        path = f"{exp_dir}/{exp_name}/{exp_name}_seed_{str(rng_key)}.npz"
        with open(path, "wb+") as f:
            jnp.savez(f, **eval_data)

        # Save sample data


if __name__ == "__main__":
    ENV_NAME = "Week6_01-v0"
    EXP_DIR = "data/200110"
    EXP_NAME = "active_ird_exp_mid"
    FN_KEYS = ["infogain", "ratiomean", "ratiomin", "random"]
    VERBOSE = False

    NUM_ACTIVE_TASKS = 36
    NUM_EVAL_TASKS = 36

    env = gym.make(ENV_NAME)
    env.reset()

    TEST_DATA = None
    data2 = load_data(env, "data/200110_test", EXP_NAME)

    TEST_DATA = data2
    data = load_data(env, EXP_DIR, EXP_NAME)

    save_data(env, data, EXP_DIR, EXP_NAME)
