import torch
import gym, os
import pytest
import numpy as onp
import rdb.envs.drive2d
from time import time
from jax import random
from numpyro.handlers import seed
from rdb.infer import *
from rdb.exps.utils import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "Week6_02-v1"
NUM_DATA = 20000
TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)
SAVE_DIR = os.path.join(data_dir(), "universal")
env = gym.make(ENV_NAME)  # Two Blockway
rng_key = random.PRNGKey(0)

## Helper functions
def _build_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in env.features_keys:
            w[key] = onp.random.random()
        weights.append(w)
    return DictList(weights)


def _make_particles(num_weights):
    env.reset()
    main_car = env.main_car
    controller, runner = build_mpc(
        env,
        main_car.cost_runtime,
        horizon=10,
        dt=env.dt,
        replan=5,
        T=10,
        engine="jax",
        method="adam",
        test_mode=True,
    )
    weights = _build_weights(num_weights)
    ps = Particles(
        rng_name="",
        rng_key=None,
        env_fn=None,
        env=env,
        controller=controller,
        runner=runner,
        normalized_key="__dummy__",
        save_name="test_particles",
        weights=weights,
        save_dir=f"{data_dir()}/test",
        weight_params={"bins": 10, "max_weights": 20},
    )
    key = random.PRNGKey(0)
    ps.update_key(key)
    return ps


## Generate weights and features sum
def generate_data():
    particles = _make_particles(NUM_DATA)
    feats_sum = particles.get_features_sum([TASK])[0]
    feats_sum = feats_sum.numpy_array().swapaxes(0, 1)  # (ndata, dim)
    weights = particles.weights.numpy_array().swapaxes(0, 1)
    ## Save
    data = dict(weights=weights, feats_sum=feats_sum)
    np.savez(os.path.join(SAVE_DIR, "training_v00.npz"), **data)


## Train network & evaluate
def load_and_train(log=True):
    global rng_key
    rng_key, rng_devel, rng_test = random.split(rng_key, 3)
    num_devel = int(0.2 * NUM_DATA)
    num_test = int(0.2 * NUM_DATA)

    data = np.load(os.path.join(SAVE_DIR, "training_v00.npz"), allow_pickle=True)
    devel_idx = random_choice(
        rng_key, np.arange(NUM_DATA), num_devel + num_test, replacement=False
    )
    test_idx = random_choice(
        rng_key, np.arange(num_devel + num_test), num_test, replacement=False
    )
    train_ones = onp.ones(NUM_DATA).astype(bool)
    train_ones[devel_idx] = False
    test_ones = onp.ones(num_devel + num_test).astype(bool)
    test_ones[test_idx] = False

    train_input = torch.from_numpy(data["weights"][train_ones]).float()
    train_output = torch.from_numpy(data["feats_sum"][train_ones]).float()
    devel_input = torch.from_numpy(
        data["weights"][np.logical_not(train_ones)][np.logical_not(test_ones)]
    ).float()
    devel_output = torch.from_numpy(
        data["feats_sum"][np.logical_not(train_ones)][np.logical_not(test_ones)]
    ).float()
    test_input = torch.from_numpy(
        data["weights"][np.logical_not(train_ones)][test_ones]
    ).float()
    test_output = torch.from_numpy(
        data["feats_sum"][np.logical_not(train_ones)][test_ones]
    ).float()

    if log:
        writer = SummaryWriter()
    D_in = train_input.shape[1]
    D_out = train_output.shape[1]
    H = 200

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    train_data = TensorDataset(train_input, train_output)
    train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    learning_rate = 1e-4
    for t in range(20000):

        for x_batch, y_batch in train_loader:
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

        train_pred = model(train_input)
        train_loss = loss_fn(train_pred, train_output)
        devel_pred = model(devel_input)
        devel_loss = loss_fn(devel_pred, devel_output)
        print("Epoch", t, "Train", train_loss.item(), "Devel", devel_loss.item())
        if log:
            writer.add_scalar("Loss/train", train_loss.item(), t)
            writer.add_scalar("Loss/devel", devel_loss.item(), t)
        model.zero_grad()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Planning Network")
    parser.add_argument("--data", action="store_true", help="Generate data")
    args = parser.parse_args()

    if args.data:
        generate_data()
    else:
        load_and_train()
