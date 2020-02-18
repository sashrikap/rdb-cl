"""Test sampling algorithms.

TODO:
[1] Convergence plots

"""


import numpy as onp
import itertools
import pytest
import numpyro
import itertools
import numpyro.distributions as dist
from rdb.infer import *
from functools import partial
from rdb.infer.dictlist import *
from rdb.infer.algos import MetropolisHasting
from numpyro.handlers import scale, condition, seed
from jax import random


class prior(object):
    def __init__(self):
        pass

    def log_prob(self, data):
        return onp.zeros(len(data))


class proposal(object):
    def __init__(self):
        self._sample_fn = None

    def update_key(self, key):
        self._sample_fn = seed(self.build_sample_fn(), key)

    def build_sample_fn(self):
        def fn(state):
            std = onp.ones_like(state) * 0.5
            return numpyro.sample("next_state", dist.Normal(state, std))

        return fn

    def __call__(self, state):
        return self._sample_fn(state)


def kernel(obs, state, tasks):
    """Dummy kernel. Posterior: mean(obs + tasks)

    Args:
        obs (ndarray): (nbatch, obs_dim, )
        state (ndarray): (nbatch, obs_dim, )
        tasks (ndarray): (ntasks, obs_dim)

    Note:
        * Performs computations on (nbatch * ntasks, )

    Output:
        out (ndarray): (nbatch, )

    """
    assert obs.shape == state.shape
    ntasks = len(tasks)
    nbatch = len(state)
    # Performs computations on each state x each task
    state_, tasks_ = zip(*list(itertools.product(state, tasks)))
    obs_, _ = zip(*list(itertools.product(obs, tasks)))
    obs_ = onp.array(obs_)
    state_ = onp.array(state_)
    tasks_ = onp.array(tasks_)
    # Sum over state_dims
    diffs_ = -onp.square(obs_ + tasks_ - state_)
    diffs_ = diffs_.reshape(nbatch * ntasks, -1).sum(axis=1)
    out = diffs_.reshape(nbatch, ntasks).sum(axis=1)

    assert out.shape == (nbatch,)
    return out


@pytest.mark.parametrize("num_chains", [1, 5, 10, 30])
def test_mh_algo(num_chains):
    state = onp.zeros(2)
    tasks = onp.array([[2, 3], [3, 4], [4, 5]])
    key = random.PRNGKey(1)
    infer = MetropolisHasting(
        None,
        kernel=kernel,
        prior=prior(),
        proposal=proposal(),
        num_samples=100,
        num_warmups=50,
        num_chains=num_chains,
    )
    infer.update_key(key)
    obs = onp.ones(2)
    samples, _ = infer.sample(obs, init_state=state, tasks=tasks)
    # Roughly [4, 5]
    print(samples.mean(axis=0))


@pytest.mark.parametrize("num_chains", [5, 10, 30])
@pytest.mark.parametrize("num_tasks", [2, 4])
def test_mh_algo_state_2d(num_chains, num_tasks):
    state = onp.zeros((5, 2))
    key = random.PRNGKey(1)
    tasks = onp.array([onp.ones((5, 2))] * num_tasks)
    infer = MetropolisHasting(
        None,
        kernel=kernel,
        prior=prior(),
        proposal=proposal(),
        num_samples=10,
        num_warmups=100,
        num_chains=num_chains,
    )
    infer.update_key(key)
    obs = onp.ones((5, 2))
    samples, _ = infer.sample(obs, init_state=state, tasks=tasks)
    print(samples.mean(axis=0))


dict_proposal = IndGaussianProposal(
    rng_key=None, normalized_key="a", feature_keys=["a", "b"], proposal_var=5
)


def dict_kernel(obs, state, tasks, xdim):
    assert obs.shape == state.shape
    weight_dim = 2
    nbatch = len(state)
    ntasks = len(tasks)
    batch_states = [state[i] for i in range(nbatch)]
    # Performs computations on each state x each task
    state_, tasks_ = zip(*list(itertools.product(batch_states, tasks)))
    obs_, _ = zip(*list(itertools.product(obs, tasks)))
    # Sum over state_dims
    tasks_ = onp.array(tasks_)
    obs_ = DictList(obs_)
    state_ = DictList(state_)
    # sum across tasks
    out = (obs_ - state_).reshape((nbatch, ntasks, xdim))
    # sum across task
    out = out.sum(axis=1)
    assert out.shape == (nbatch, xdim)
    assert out.num_keys == weight_dim
    # sum across xdim
    out = out.sum(axis=1).sum_values()
    return out


class dict_proposal(object):
    def __init__(self):
        self._sample_fn = None

    def update_key(self, key):
        self._sample_fn = seed(self.build_sample_fn(), key)

    def build_sample_fn(self):
        def fn(state):
            std = onp.ones_like(state) * 0.5
            out = {}
            for key, val in state.items():
                out[key] = numpyro.sample("next_state", dist.Normal(val, std))
            return DictList(out)

        return fn

    def __call__(self, state):
        return self._sample_fn(state)


class dict_prior(object):
    def __init__(self):
        pass

    def log_prob(self, data):
        return onp.zeros(len(data))


@pytest.mark.parametrize("xdim", [1, 5])
@pytest.mark.parametrize("num_chains", [1, 2, 10, 30])
@pytest.mark.parametrize("num_tasks", [2, 4])
def test_mh_algo_dict(xdim, num_chains, num_tasks):
    num_tasks = 3
    weight_dim = 2
    num_samples = 100
    key = random.PRNGKey(1)
    obs = DictList({"a": onp.ones(xdim), "b": 3 * onp.ones(xdim)})
    tasks = [1, 2, 3]
    state = DictList({"a": onp.zeros(xdim), "b": 3 * onp.zeros(xdim)})
    infer = MetropolisHasting(
        None,
        kernel=partial(dict_kernel, xdim=xdim),
        prior=dict_prior(),
        proposal=dict_proposal(),
        num_samples=num_samples,
        num_warmups=100,
        num_chains=num_chains,
        use_dictlist=True,
    )
    infer.update_key(key)
    samples, info = infer.sample(obs, init_state=state, tasks=tasks)
    assert samples.shape == (num_samples, xdim)
    assert samples.num_keys == weight_dim
    for samples_i in info["all_chains"]:
        assert samples_i.num_keys == weight_dim
        assert samples_i.shape[1] == xdim
    visualize_chains(
        info["all_chains"],
        info["rates"],
        fig_dir=f"data/test/mcmc",
        title=f"test_mcmc_xdim_{xdim}_chains_{num_chains}_tasks_{num_tasks}",
    )
