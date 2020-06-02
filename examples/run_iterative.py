import gym
import copy
import argparse
import rdb.envs.drive2d
from jax import random
from rdb.infer.utils import *
from functools import partial
from rdb.optim.mpc import build_mpc
from rdb.exps.iterative_ird import run_iterative
from rdb.exps.utils import load_params, examples_dir, data_dir


def run():
    experiment = run_iterative(evaluate=evaluate, gcp_mode=GCP_MODE)
    for random_key in RANDOM_KEYS:
        rng_key = random.PRNGKey(random_key)
        experiment.update_key(rng_key)
        # experiment.evaluate()
        experiment.add_evaluate_obs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--GCP_MODE", action="store_true")
    args = parser.parse_args()
    GCP_MODE = args.GCP_MODE

    ## Runs evaluation by default
    evaluate = True

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params(f"{examples_dir()}/params/iterative_template.yaml")
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/iterative_params.yaml")

    locals().update(PARAMS)
    run()
