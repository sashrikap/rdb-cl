import torch
import gym, os
import pytest
import numpy as onp
import rdb.envs.drive2d
import numpy.random as npr
from time import time
from jax import random
from numpyro.handlers import seed
from rdb.infer import *
from rdb.exps.utils import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer.universal import *
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from jax.experimental import stax
from jax import jit, grad, random
from jax.experimental import optimizers

## Helper functions
def _env_fn(env_name=None):
    import gym, rdb.envs.drive2d

    if env_name is None:
        env_name = ENV_NAME
    env = gym.make(env_name)
    env.reset()
    return env


def _build_weights(num_weights):
    weights = []
    for _ in range(num_weights):
        w = {}
        for key in TRUE_W.keys():
            if key in FEATS_KEYS:
                w[key] = onp.exp((onp.random.random() * 2 - 1) * MAX_W)
            else:
                w[key] = TRUE_W[key]
        weights.append(w)
    return DictList(weights)


def _controller_fn(env, name=""):
    controller, runner = build_mpc(
        env,
        env.main_car.cost_runtime,
        dt=env.dt,
        name=name,
        replan=5,
        T=10,
        engine="jax",
        method="adam",
    )
    return controller, runner


def _make_particles(num_weights):
    env.reset()
    controller, runner = _controller_fn(env)
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
        weight_params={"bins": 10, "max_weights": MAX_W},
    )
    key = random.PRNGKey(0)
    ps.update_key(key)
    return ps


## Generate weights and features sum
def generate_data():
    particles = _make_particles(NUM_DATA)
    ## Weights
    weights = particles.weights
    log_weights = particles.weights.log()
    keys = list(log_weights.keys())
    for key in keys:
        if key not in FEATS_KEYS:
            del log_weights[key]
    weights = weights.numpy_array().swapaxes(0, 1)
    log_weights = log_weights.numpy_array().swapaxes(0, 1)
    feats_sum = particles.get_features_sum([TASK])[0]
    feats_sum = feats_sum.numpy_array().swapaxes(0, 1)  # (ndata, dim)
    ## Save
    data = dict(weights=log_weights, feats_sum=feats_sum, feats_keys=FEATS_KEYS)
    np.savez(os.path.join(SAVE_DIR, DATA_FILE), **data)


def _load_data(use_torch=False):
    global rng_key
    rng_key, rng_devel, rng_test = random.split(rng_key, 3)
    num_devel = int(0.1 * NUM_DATA)
    num_test = int(0.1 * NUM_DATA)

    data = np.load(os.path.join(SAVE_DIR, DATA_FILE), allow_pickle=True)
    devel_idx = random.choice(
        rng_devel, np.arange(NUM_DATA), (num_devel + num_test,), replace=False
    )
    test_idx = random.choice(
        rng_test, np.arange(num_devel + num_test), (num_test,), replace=False
    )
    train_ones = onp.ones(NUM_DATA).astype(bool)
    train_ones[devel_idx] = False
    test_ones = onp.ones(num_devel + num_test).astype(bool)
    test_ones[test_idx] = False

    train_input = data["weights"][train_ones].astype(float)
    train_output = data["feats_sum"][train_ones].astype(float)
    devel_input = data["weights"][np.logical_not(train_ones)][
        np.logical_not(test_ones)
    ].astype(float)
    devel_output = data["feats_sum"][np.logical_not(train_ones)][
        np.logical_not(test_ones)
    ].astype(float)
    test_input = data["weights"][np.logical_not(train_ones)][test_ones].astype(float)
    test_output = data["feats_sum"][np.logical_not(train_ones)][test_ones].astype(float)
    if use_torch:
        train_input = torch.from_numpy(train_input)
        train_output = torch.from_numpy(train_output)
        devel_input = torch.from_numpy(devel_input)
        devel_output = torch.from_numpy(devel_output)
        test_input = torch.from_numpy(test_input)
        test_output = torch.from_numpy(test_output)

    print(f"Loaded train input: {train_input.shape}")
    print(f"Loaded train output: {train_output.shape}")
    print(f"Loaded devel input: {devel_input.shape}")
    print(f"Loaded test input: {test_input.shape}")
    return train_input, train_output, devel_input, devel_output, test_input, test_output


## Train network & evaluate
def load_and_train(log=True):
    train_input, train_output, devel_input, devel_output, test_input, test_output = (
        _load_data()
    )

    if log:
        writer = SummaryWriter()

    num_epochs = int(5e4)
    batch_size = 1024
    num_batches, leftover = divmod(train_input.shape[0], batch_size)
    num_batches += bool(leftover)
    itercount = itertools.count()

    # Optimization specific
    init, predict = create_model(train_input.shape[1], train_output.shape[1])
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
    init_key = random.PRNGKey(0)
    _, init_params = init(init_key, (-1, len(FEATS_KEYS)))
    opt_state = opt_init(init_params)

    rand_key = random.PRNGKey(0)

    @jax.jit
    def loss_fn(params, batch):
        nonlocal rand_key
        inputs, targets = batch
        key, rand_key = random.split(rand_key)
        preds = predict(params, inputs, rng=key)
        return np.mean(np.sum(np.square(preds - targets), axis=1))

    @jax.jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss_fn)(params, batch), opt_state)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(train_input.shape[0])
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_input[batch_idx], train_output[batch_idx]

    batches = data_stream()
    os.makedirs(os.path.join(SAVE_DIR, MODEL_NAME), exist_ok=True)

    for epoch in range(num_epochs):

        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        params = get_params(opt_state)
        train_loss = loss_fn(params, (train_input, train_output))
        if epoch % TEST_PER == 0:
            test_loss = loss_fn(params, (test_input, test_output))
            print(
                f"Epoch {epoch} ({epoch_time:0.2f}s) Training loss {train_loss:0.5f} Test set loss {test_loss:0.5f}"
            )
        else:
            print(f"Epoch {epoch} ({epoch_time:0.2f}s) Training loss {train_loss:0.5f}")

        if log:
            writer.add_scalar("Loss/train", train_loss.item(), epoch)
            writer.add_scalar("Loss/test", test_loss.item(), epoch)

        if epoch % SAVE_PER == 0:
            np.savez(
                os.path.join(SAVE_DIR, MODEL_NAME, f"{MODEL_NAME}_{epoch:05}.npz"),
                params=params,
                feats_keys=FEATS_KEYS,
                input_dim=train_input.shape[1],
                output_dim=train_output.shape[1],
            )


# Load trained network and run inference
def run_inference():
    global rng_key
    rng_key, rng_designer = random.split(rng_key)
    data = np.load(os.path.join(SAVE_DIR, DATA_FILE), allow_pickle=True)
    train_input = torch.from_numpy(data["weights"]).float()
    train_output = torch.from_numpy(data["feats_sum"]).float()

    for epoch in [200, 500, 1000, 2000, 4000, 6000]:
        model = create_model(train_input, train_output)
        model.load_state_dict(
            torch.load(os.path.join(SAVE_DIR, MODEL_NAME, f"{MODEL_NAME}_{epoch:05}"))
        )

        def prior_fn(name="", feature_keys=WEIGHT_PARAMS["feature_keys"]):
            return LogUniformPrior(
                normalized_key="__dummy__",
                feature_keys=FEATS_KEYS,
                log_max=MAX_W,
                # default=TRUE_W,
                name="universal_inference_prior",
            )

        designer = Designer(
            env_fn=_env_fn,
            controller_fn=_controller_fn,
            prior_fn=prior_fn,
            prior_keys=FEATS_KEYS,
            weight_params=WEIGHT_PARAMS,
            normalized_key="__dummy__",
            save_root=f"{SAVE_DIR}/{MODEL_NAME}",
            exp_name=f"universal_inference_{epoch:04}",
            **DESIGNER_ARGS,
        )
        designer.update_key(rng_key)
        designer.true_w = TRUE_W
        proxies = designer.simulate(
            onp.array([TASK]),
            save_name=f"universal_inference_{DESIGNER_ARGS['beta']}",
            universal_model=model,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Planning Network")
    parser.add_argument("--data", action="store_true", help="Generate data")
    parser.add_argument("--infer", action="store_true", help="Generate data")
    args = parser.parse_args()

    ENV_NAME = "Week6_02-v1"
    NUM_DATA = 40000
    # TASK = (-0.7, -0.7, 0.13, 0.4, -0.13, 0.4)
    TASK = (-0.5, -0.5, -0.08, 0.5, 0.12, -0.9)
    SAVE_DIR = os.path.join(data_dir(), "universal")
    env = _env_fn(ENV_NAME)  # Two Blockway
    rng_key = random.PRNGKey(0)
    TRUE_W = {
        "dist_cars": 1.0,
        "dist_lanes": 1.25,
        "dist_fences": 2.675,
        "dist_objects": 1.125,
        "speed_80": 2.5,
        "control": 2.0,
    }
    # FEATS_KEYS = ["dist_cars", "dist_objects"]
    FEATS_KEYS = [
        "dist_cars",
        "dist_lanes",
        "dist_fences",
        "dist_objects",
        "speed_80",
        "control",
    ]
    MAX_W = 15
    SAVE_PER = 100
    TEST_PER = 10
    MODEL_NAME = "model_v01"
    DATA_FILE = "training_v01.npz"
    ## v00: ["dist_cars", "dist_objects"], 40000
    WEIGHT_PARAMS = {
        "normalized_key": "__dummy__",
        "viz_normalized_key": "__dummy__",
        "max_weights": 15.0,
        "bins": 200,
        "feature_keys": FEATS_KEYS
        # ["dist_cars", "dist_lanes", "dist_fences", "dist_objects", "speed_80", "control"]
    }
    DESIGNER_ARGS = {
        "task_method": "mean",
        "num_normalizers": 1000,
        "sample_method": "MH",
        # design_mode: independent
        "design_mode": "joint",
        "select_mode": "mean",
        "sample_init_args": {
            # proposal_var: 0.1
            "proposal_var": 2,
            # proposal_var: 1.5 # beta = 0.1
            # proposal_var: 2 # beta = 0.05
            # proposal_var: 1
            "max_val": 15.0,
        },
        "proposal_decay": 0.7,
        "sample_args": {"num_samples": 2000, "num_warmup": 25, "num_chains": 1},
        "beta": 1,
    }

    if args.data:
        generate_data()
    elif args.infer:
        run_inference()
    else:
        load_and_train()


# def load_and_train_torch(log=True):
#     train_input, train_output, devel_input, devel_output, test_input, test_output = _load_data(use_torch=True)

#     if log:
#         writer = SummaryWriter()

#     num_epochs = int(5e4)
#     batch_size = 1024
#     num_feats = len(env.feature_keys)

#     model = create_model(len(feats_keys), num_feats)
#     train_data = TensorDataset(train_input, train_output)
#     train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#     loss_fn = torch.nn.MSELoss(reduction="mean")
#     learning_rate = 1e-4
#     os.makedirs(os.path.join(SAVE_DIR, MODEL_NAME), exist_ok=True)
#     for t in range(num_epochs):

#         for x_batch, y_batch in train_loader:
#             # x_batch = x_batch.to(device)
#             # y_batch = y_batch.to(device)
#             y_pred = model(x_batch)
#             loss = loss_fn(y_pred, y_batch)
#             model.zero_grad()
#             loss.backward()
#             with torch.no_grad():
#                 for param in model.parameters():
#                     param -= learning_rate * param.grad

#         train_pred = model(train_input)
#         train_loss = loss_fn(train_pred, train_output)
#         devel_pred = model(devel_input)
#         devel_loss = loss_fn(devel_pred, devel_output)
#         print(f"Epoch {t} Train {train_loss.item():.6f} Devel {devel_loss.item():.6f}")
#         if log:
#             writer.add_scalar("Loss/train", train_loss.item(), t)
#             writer.add_scalar("Loss/devel", devel_loss.item(), t)

#         if t % SAVE_PER == 0:
#             torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME, f"{MODEL_NAME}_{t:05}"))
#         model.zero_grad()


# def _create_model_torch(train_input, train_output):
#     D_in = train_input.shape[1]
#     D_out = train_output.shape[1]
#     H = 200

#     model = torch.nn.Sequential(
#         torch.nn.Linear(D_in, H),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H, H),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H, H),
#         torch.nn.ReLU(),
#         torch.nn.Linear(H, D_out),
#     )
#     return model
