# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import os
import jax
import pytest
import numpyro
import numpy as onp
import jax.numpy as np
import numpyro.distributions as dist
from jax.test_util import check_close
from numpy.testing import assert_allclose
from jax import jit, pmap, random, vmap
from jax.lib import xla_bridge
from jax.scipy.special import logit
from numpyro.distributions import constraints
from numpyro.infer import HMC, MCMC, NUTS, SA, MH
from numpyro.infer.mcmc import hmc, _get_proposal_loc_and_scale, _numpy_delete, mh_draws
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect, control_flow_prims_disabled
from numpyro.infer.mcmc import hmc, mh
from numpyro.handlers import seed, substitute, trace
from numpyro.distributions.distribution import enable_validation, validation_enabled


def test_substitute():
    # numpyro.enable_validation()
    def model_normal():
        numpyro.sample("a", dist.Normal(0, 1))

    def model_tnormal():
        numpyro.sample("a", dist.TruncatedNormal(-1, 0, 1))

    def model_uniform():
        numpyro.sample("a", dist.Uniform(0.0, 1.0))

    model_normal = seed(model_normal, random.PRNGKey(0))
    model_tnormal = seed(model_tnormal, random.PRNGKey(0))
    model_uniform = seed(model_uniform, random.PRNGKey(0))
    trace(substitute(model_uniform, {"a": -1})).get_trace()
    trace(substitute(model_normal, {"a": -1})).get_trace()
    trace(substitute(model_tnormal, {"a": -1})).get_trace()
    # import pdb; pdb.set_trace()
    # assert exec_trace['a']['value'] == -1
    # exec_trace = trace(substitute(model, {'a': 0.5})).get_trace()


def test_log_density():
    def model(labels):
        coefs = numpyro.sample("coefs", dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = numpyro.deterministic("logits", np.sum(coefs * data, axis=-1))
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)


num_chains = 1
numpyro.set_host_device_count(num_chains)


def test_mh_func():
    true_coefs = np.array([1.0, 2.0, 3.0])
    data = random.normal(random.PRNGKey(2), (2000, 3))
    dim = 3
    labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(
        random.PRNGKey(3)
    )

    def model(data, labels):
        coefs_mean = np.zeros(dim)
        coefs = numpyro.sample("beta", dist.Normal(coefs_mean, np.ones(3)))
        intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
        return numpyro.sample(
            "y", dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels
        )

    kernel = MH(model, proposal_var=0.05)
    # kernel = HMC(model)
    # init_params = {"beta": np.zeros(dim), "intercept": 10.}
    mcmc = MCMC(
        kernel,
        1000,
        10000,
        num_chains=num_chains,
        # chain_method='vectorized',
        progress_bar=True,
    )
    # with jax.disable_jit(), control_flow_prims_disabled():
    mcmc.run(
        random.PRNGKey(0), data=data, labels=labels, extra_fields=["mean_accept_prob"]
    )
    samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    extra_fields["mean_accept_prob"][:, -1]
    mcmc.print_summary()
    print(np.mean(samples["beta"][0], axis=0))  # doctest: +SKIP
    print(np.mean(samples["intercept"][0]))  # doctest: +SKIP
    # import pdb; pdb.set_trace()
    # [0.9153987 2.0754058 2.9621222]


def test_beta_bernoulli_x64():
    warmup_steps, num_samples = (1000, 10000)

    def model(data):
        alpha = np.array([1.1, 1.1])
        beta = np.array([1.1, 1.1])
        p_latent = numpyro.sample("p_latent", dist.Beta(alpha, beta))
        numpyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = np.array([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(random.PRNGKey(1), (1000, 2))
    kernel = MH(model=model, proposal_var=0.001)
    # kernel = HMC(model=model)
    init_params = {"p_latent": np.array([0.9, 0.1])}
    mcmc = MCMC(
        kernel, num_warmup=warmup_steps, num_samples=num_samples, progress_bar=True
    )
    mcmc.run(random.PRNGKey(2), data=data, init_params=init_params)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    # assert_allclose(np.mean(samples['p_latent'], 0), true_probs, atol=0.05)
    print(np.mean(samples["p_latent"], 0))


def test_random():
    p = [0.5, 0.7]

    @jit  # consider keeping or removing this jit
    def f():
        out = []
        rng = random.PRNGKey(0)

        def _cond_fn(val):
            rng, out, i = val
            return i < 3

        def _body_fn(val):
            rng, out, i = val
            rng, rng_input = random.split(rng)
            out += random.uniform(rng_input, (5,))
            return (rng, out, i + 1)

        return jax.lax.while_loop(_cond_fn, _body_fn, (rng, np.zeros(5), 0))

    print(f())


# >>> true_coefs = np.array([1., 2., 3.])
# >>> data = random.normal(random.PRNGKey(2), (2000, 3))
# >>> dim = 3
# >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample(random.PRNGKey(3))
# >>>
# >>> def model(data, labels):
# ...     coefs_mean = np.zeros(dim)
# ...     coefs = numpyro.sample('beta', dist.Normal(coefs_mean, np.ones(3)))
# ...     intercept = numpyro.sample('intercept', dist.Normal(0., 10.))
# ...     return numpyro.sample('y', dist.Bernoulli(logits=(coefs * data + intercept).sum(-1)), obs=labels)
# >>>
# >>> init_params, potential_fn, constrain_fn = initialize_model(random.PRNGKey(0),
# ...                                                            model, model_args=(data, labels,))
# >>> init_kernel, sample_kernel = hmc(potential_fn, algo='NUTS')
# >>> hmc_state = init_kernel(init_params,
# ...                         trajectory_length=10,
# ...                         num_warmup=300)
# >>> samples = fori_collect(0, 500, sample_kernel, hmc_state,
# ...                        transform=lambda state: constrain_fn(state.z))
# >>> print(np.mean(samples['beta'], axis=0))  # doctest: +SKIP
# [0.9153987 2.0754058 2.9621222]


@pytest.mark.parametrize("kernel_cls", [MH])
def test_logistic_regression_x64(kernel_cls):
    N, dim = 3000, 3
    warmup_steps, num_samples = (100000, 100000) if kernel_cls is SA else (1000, 8000)
    data = random.normal(random.PRNGKey(0), (N, dim))
    true_coefs = np.arange(1.0, dim + 1.0)
    logits = np.sum(true_coefs * data, axis=-1)
    labels = dist.Bernoulli(logits=logits).sample(random.PRNGKey(1))

    def model(labels):
        coefs = numpyro.sample("coefs", dist.Normal(np.zeros(dim), np.ones(dim)))
        logits = numpyro.deterministic("logits", np.sum(coefs * data, axis=-1))
        return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

    kernel = kernel_cls(model=model)
    mcmc = MCMC(kernel, warmup_steps, num_samples, progress_bar=False)
    mcmc.run(random.PRNGKey(2), labels)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    assert samples["logits"].shape == (num_samples, N)
    assert_allclose(np.mean(samples["coefs"], 0), true_coefs, atol=0.22)

    if "JAX_ENABLE_X64" in os.environ:
        assert samples["coefs"].dtype == np.float64
