"""Universal framework for different symbolic equations.

"""
import jax.numpy as np
import numpy as onp
import itertools
import numpyro
import torch
import time
import jax
import numpy.random as npr
import numpyro.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from jax.experimental import optimizers
from numpyro.infer import MCMC, NUTS
from collections import OrderedDict
from jax import jit, grad, random
from jax.experimental import stax
from jax import random


def train_model(data, log=True):
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]

    D_in = train_x.shape[1]
    D_out = train_y.shape[1]
    H = 200
    # num_epochs = int(5)
    num_epochs = int(50)
    batch_size = 1024
    num_complete_batches, leftover = divmod(train_x.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    ## RUNNING OPTIMIZATION ##
    init, predict = stax.serial(
        stax.Dense(H, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(H, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(H, W_init=stax.randn()),
        stax.Relu,
        stax.Dense(D_out, W_init=stax.randn()),
    )
    opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
    init_key = random.PRNGKey(0)
    _, init_params = init(init_key, (-1, 2))
    opt_state = opt_init(init_params)

    itercount = itertools.count()

    def loss(params, batch):
        inputs, targets = batch
        preds = predict(params, inputs)
        return np.mean(np.sum(np.square(preds - targets), axis=1))

    @jax.jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(train_x.shape[0])
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_x[batch_idx], train_y[batch_idx]

    batches = data_stream()
    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time
        params = get_params(opt_state)
        train_loss = loss(params, (train_x, train_y))
        test_loss = loss(params, (test_x, test_y))
        print(
            f"Epoch {epoch} ({epoch_time:0.2f}s) Training loss {train_loss:0.5f} Test set loss {test_loss:0.5f}"
        )

    def nn_model(data):
        return predict(params, data)

    return nn_model


def compare_inference(nn_model, forward_fn, example_type):
    if example_type == "parabola_sol":
        """ p(a, b) = [argmin_x (ax^2 + bx)]^2 """
        prior_params = OrderedDict({"a": (0.5, 1.0), "b": (-1.0, 1.0)})
        log_prob_fn = lambda sol: np.log(sol ** 2)
        xlim, ylim = (0.5, 1), (-1, 1)
    elif example_type == "parabola_val":
        """ p(a, b) = [0.5 + min_x (ax^2 + bx)]^2 """
        prior_params = OrderedDict({"a": (0.5, 1.0), "b": (-1.0, 1.0)})
        log_prob_fn = lambda sol: np.log((0.5 + sol) ** 2)
        xlim, ylim = (0.5, 1), (-1, 1)

    def universal_model():
        prior = []
        for key, val in prior_params.items():
            prior.append(numpyro.sample(key, dist.Uniform(*val)))
        val = nn_model(np.array(prior))
        numpyro.factor("pred_log_prob", log_prob_fn(val))

    num_warmup = 5000
    num_sample = 10000
    rng_key = random.PRNGKey(0)

    start = time.time()
    kernel = NUTS(universal_model)
    mcmc = MCMC(kernel, num_warmup, num_sample, num_chains=1, progress_bar=True)
    mcmc.run(rng_key)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    samples = mcmc.get_samples()
    plot_samples(
        samples,
        f"data/universal/example_{example_type}_nn.png",
        xlim=xlim,
        ylim=ylim,
        keys=list(prior_params.keys()),
    )

    def forward_model():
        prior = []
        for key, val in prior_params.items():
            prior.append(numpyro.sample(key, dist.Uniform(*val)))
        val = forward_fn(np.array(prior))
        numpyro.factor("pred_log_prob", log_prob_fn(val))

    start = time.time()
    kernel = NUTS(forward_model)
    mcmc = MCMC(kernel, num_warmup, num_sample, num_chains=1, progress_bar=True)
    mcmc.run(rng_key)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    samples = mcmc.get_samples()
    plot_samples(
        samples,
        f"data/universal/example_{example_type}_gt.png",
        xlim=xlim,
        ylim=ylim,
        keys=list(prior_params.keys()),
    )


def plot_samples(samples, filename, xlim, ylim, keys):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    ms = 1
    xs = samples[keys[0]]
    ys = samples[keys[1]]
    plt.scatter(xs, ys, s=[1 for _ in range(len(xs))])
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(filename)


def build_function(example_type):
    """Find forward_fn, and sample_fn.

    Note:
        forward_fn: fn(x) -> y (float)
        sample_fn: fn(n) -> {
            "train_x": (ndata, input_dims),
            "train_y": (ndata, solution_dims),
            "test_x": (ndata, input_dims),
            "test_y": (ndata, solution_dims),
        }

    """

    if example_type == "parabola_sol":

        def _sample_fn(rng_key, num_train, num_test):
            rng_train, rng_test = random.split(rng_key, 2)
            a = onp.random.uniform(0.5, 1, (num_train, 1))
            b = onp.random.uniform(-1, 1, (num_train, 1))
            train_x = onp.concatenate([a, b], axis=1)  # (a, b)
            train_y = (-0.5 * train_x[:, 1] / train_x[:, 0])[:, None]
            a = onp.random.uniform(0.5, 1, (num_test, 1))
            b = onp.random.uniform(-1, 1, (num_test, 1))
            test_x = onp.concatenate([a, b], axis=1)  # (a, b)
            test_y = (-0.5 * test_x[:, 1] / test_x[:, 0])[:, None]
            return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

        def _forward_fn(x):
            return -0.5 * x[1] / x[0]

    elif example_type == "parabola_val":

        def _sample_fn(rng_key, num_train, num_test):
            rng_train, rng_test = random.split(rng_key, 2)
            a = onp.random.uniform(0.5, 1, (num_train, 1))
            b = onp.random.uniform(-1, 1, (num_train, 1))
            train_x = onp.concatenate([a, b], axis=1)  # (a, b)
            train_y = (-0.25 * np.square(train_x[:, 1]) / train_x[:, 0])[:, None]
            a = onp.random.uniform(0.5, 1, (num_test, 1))
            b = onp.random.uniform(-1, 1, (num_test, 1))
            test_x = onp.concatenate([a, b], axis=1)  # (a, b)
            test_y = (-0.25 * np.square(test_x[:, 1]) / test_x[:, 0])[:, None]
            return dict(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

        def _forward_fn(x):
            return -0.25 * x[1] ** 2 / x[0]

    else:
        raise NotImplementedError

    return _sample_fn, _forward_fn


def main(example_type):
    print(f"Running example of type {example_type}")
    num_train = 20000
    num_test = 2000
    rng_seed = 0

    rng_key = random.PRNGKey(rng_seed)
    sample_fn, forward_fn = build_function(example_type)
    rng_key, rng_data = random.split(rng_key)
    data = sample_fn(rng_data, num_train, num_test)
    model = train_model(data)

    compare_inference(model, forward_fn, example_type)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Planning Examples")
    parser.add_argument(
        "--type",
        type=str,
        choices=["parabola_sol", "parabola_val"],
        help="Example type",
    )
    args = parser.parse_args()
    assert args.type is not None

    main(example_type=args.type)
