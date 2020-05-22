import gym
import copy
import rdb.envs.drive2d
from jax import random
from rdb.infer.utils import *
from functools import partial
from rdb.optim.mpc import build_mpc
from rdb.exps.active_user import run_experiment_server


if __name__ == "__main__":
    test_mode = False

    experiment = run_experiment_server(test_mode=test_mode)
    experiment.run()
