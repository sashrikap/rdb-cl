"""MCMC Inference Algorithms.

Given p(y | theta), estimate p(theta | y)

Note
    * Metropolis-Hasting written as plug-in for numpyro
    * Allows jit=True/False in forward model. jit=False is useful for
      designer model with inner-optimization.

Used for:
    * Active Inverse Reward Design Experiments
    * Divide and Conquer IRD

Credits:
    * Jerry Z. He 2019-2020

"""

import numpyro.distributions as dist
import jax.numpy as jnp
import numpy as onp
import numpyro
import copy
import jax
import os
from numpyro.handlers import block, seed, substitute, trace, scale, condition
from jax import jit, lax, pmap, random, vmap, device_get, device_put
from functools import partial
from numpyro.infer.mcmc import MCMCKernel, _collect_fn
from jax.flatten_util import ravel_pytree
from rdb.infer.dictlist import DictList
from abc import ABC, abstractmethod
from rdb.exps.utils import Profiler
from collections import namedtuple
from tqdm.auto import tqdm, trange
from numpyro.infer.util import (
    init_to_uniform,
    get_potential_fn,
    find_valid_initial_params,
    log_density,
)
from numpyro.util import (
    not_jax_tracer,
    while_loop,
    cond,
    fori_loop,
    identity,
    not_jax_tracer,
    cached_by,
)
from jax import random


"""
A :func:`~collections.namedtuple` consisting of the following fields:

 - **curr_state** - current state.
 - **rng_key** - random number generator seed used for the iteration.
"""

MHState = namedtuple(
    "MHState",
    [
        "i",
        "z",
        "curr_log_prob",
        "num_steps",
        "mean_accept_prob",
        "rng_key",
        "diverging",
    ],
)


# =========================================================================
# ======================== Numpyro MCMC Algorithm =========================
# =========================================================================


def mh_draws(state, proposal_var, rng_key=None):
    """
    Draw next state for metropolis hasting.

    :param jax.random.PRNGKey rng_key: source of the randomness, defaults to `jax.random.PRNGKey(0)`.
    """
    rng_key = random.PRNGKey(0) if rng_key is None else rng_key

    sample_val = (
        lambda mean: random.normal(rng_key, mean.shape) * jnp.sqrt(proposal_var) + mean
    )

    return sample_val(state)


def mh(model, proposal_var, jit=True):
    r"""
    Metropolis Hasting inference.

    **References:**

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param potential_fn_gen: Python callable that when provided with model
        arguments / keyword arguments returns `potential_fn`. This
        may be provided to do inference on the same model with changing data.
        If the data shape remains the same, we can compile `sample_kernel`
        once, and use the same for multiple inference runs.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum. If not provided, the default is
        euclidean kinetic energy.
    :return: a tuple of callables (`init_kernel`, `sample_kernel`), the first
        one to initialize the sampler, and the second one to generate samples
        given an existing one.

    .. warning::
        Instead of using this interface directly, we would highly recommend you
        to use the higher level :class:`numpyro.infer.MCMC` API instead.

    """

    mh_proposal_var = None
    wa_steps = None

    def init_kernel(
        init_params,
        num_warmup,
        model_args=(),
        model_kwargs=None,
        rng_key=random.PRNGKey(0),
    ):
        """
        Initializes the MH sampler.

        :param init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param int num_warmup: Number of warmup steps; samples generated
            during warmup are discarded.
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :param jax.random.PRNGKey rng_key: random key to be used as the source of
            randomness.

        """
        nonlocal mh_proposal_var, wa_steps
        wa_steps = num_warmup
        mh_proposal_var = proposal_var
        z = init_params
        kwargs = {} if model_kwargs is None else model_kwargs
        init_log_prob, _ = log_density(model, model_args, model_kwargs, init_params)
        mh_state = MHState(0.0, z, init_log_prob, 0.0, 0.0, rng_key, False)
        return device_put(mh_state)

    def _next(curr_state, curr_log_prob, model_args, model_kwargs, rng_key):

        curr_flat, unravel_fn = ravel_pytree(curr_state)
        next_log_prob, next_flat = -jnp.inf, curr_flat
        # import pdb; pdb.set_trace()
        init_val = (next_flat, next_log_prob, False, 0.0, rng_key)

        def _cond_fn(val):
            """while _cond_fn()."""
            (_, _, accept, _, _) = val
            return jnp.logical_not(accept)

        def _step_mh(rng_key):
            nonlocal curr_flat
            next_flat = mh_draws(curr_flat, mh_proposal_var, rng_key)
            next_state = unravel_fn(next_flat)
            next_log_prob, _ = log_density(model, model_args, model_kwargs, next_state)
            # curr_log_prob, _ = log_density(model, model_args, model_kwargs, curr_state)
            # print(next_log_prob, curr_log_prob)
            # print("mcmc diff", next_log_prob - curr_log_prob)
            # import pdb; pdb.set_trace()
            return next_flat, next_log_prob

        def _body_fn(val):
            (_, _, _, n, rng_key) = val
            rng_key, rng_mh_sample, rng_mh_step = random.split(rng_key, 3)
            next_flat, next_log_prob = _step_mh(rng_mh_step)
            # Forward logit minus last logit
            accept_log_prob = next_log_prob - curr_log_prob
            coin_flip_prob = jnp.log(
                random.uniform(rng_mh_sample, accept_log_prob.shape)
            )
            accept = coin_flip_prob < accept_log_prob
            # print("curr", curr_log_prob, "next", next_log_prob, "accept", accept_log_prob, accept)
            # import pdb; pdb.set_trace()
            return (next_flat, next_log_prob, accept, n + 1.0, rng_key)

        if jit:
            # Regular Numpyro workflow
            terminal_val = while_loop(_cond_fn, _body_fn, init_val)
        else:
            # Avoid JIT-compile for forward model that has inner-optimization (e.g. designer)
            curr_val = init_val
            while _cond_fn(curr_val):
                curr_val = _body_fn(curr_val)
            terminal_val = curr_val
        (next_flat, next_log_prob, _, n, rng_key) = terminal_val
        next_state = unravel_fn(next_flat)
        # import pdb; pdb.set_trace()
        # print(next_state)
        return next_state, next_log_prob, n, 1.0 / n

    def sample_kernel(mh_state, model_args=(), model_kwargs=None):
        """
        Given an existing :data:`~numpyro.infer.mcmc.MHState`, run MH with fixed (possibly adapted)
        step size and return a new :data:`~numpyro.infer.mcmc.MHState`.

        :param mh_state: Current sample (and associated state).
        :param tuple model_args: Model arguments if `potential_fn_gen` is specified.
        :param dict model_kwargs: Model keyword arguments if `potential_fn_gen` is specified.
        :return: new proposed :data:`~numpyro.infer.mcmc.MHState` from simulating
            Hamiltonian dynamics given existing state.

        """
        model_kwargs = {} if model_kwargs is None else model_kwargs
        rng_key, rng_next_key = random.split(mh_state.rng_key)

        next_state, next_log_prob, num_steps, accept_prob = _next(
            mh_state.z, mh_state.curr_log_prob, model_args, model_kwargs, rng_next_key
        )

        itr = mh_state.i + 1
        n = jnp.where(mh_state.i < wa_steps, itr, itr - wa_steps)
        mean_accept_prob = (
            mh_state.mean_accept_prob + (accept_prob - mh_state.mean_accept_prob) / n
        )

        return MHState(
            itr, next_state, next_log_prob, num_steps, mean_accept_prob, rng_key, False
        )

    # Make `init_kernel` and `sample_kernel` visible from the global scope once
    # `mh` is called for sphinx doc generation.
    if "SPHINX_BUILD" in os.environ:
        mh.init_kernel = init_kernel
        mh.sample_kernel = sample_kernel

    return init_kernel, sample_kernel


class MH(MCMCKernel):
    """
    Metropolis Hasting inference.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param float proposal_var: Proposal variance
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(
        self, model=None, proposal_var=None, init_strategy=init_to_uniform(), jit=True
    ):
        self._model = model
        self._proposal_var = proposal_var
        self._algo = "MH"
        # Set on first call to init
        self._postprocess_fn = None
        self._sample_fn = None
        self._init_fn = None
        self._init_strategy = init_strategy
        # JIT-Compile self._model
        self._jit = jit

    def _init_state(self, rng_key, model_args, model_kwargs):
        self._init_fn, self._sample_fn = mh(
            self._model, proposal_var=self._proposal_var, jit=self._jit
        )

    @property
    def sample_field(self):
        return "z"

    @property
    def model(self):
        return self._model

    def init(self, rng_key, num_warmup, init_params, model_args=(), model_kwargs={}):
        assert init_params, "Must provide initial parameters"
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1
            )
            # we need only a single key for initializing PE / constraints fn
            rng_key_init_model = rng_key_init_model[0]
        if not self._init_fn:
            self._init_state(rng_key_init_model, model_args, model_kwargs)
        # Find valid initial params
        if self._model and not init_params:
            init_params, is_valid = find_valid_initial_params(
                rng_key,
                self._model,
                init_strategy=self._init_strategy,
                param_as_improper=True,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
            if not_jax_tracer(is_valid):
                if device_get(~np.all(is_valid)):
                    raise RuntimeError(
                        "Cannot find valid initial parameters. "
                        "Please check your model again."
                    )
        mh_init_fn = lambda init_params, rng_key: self._init_fn(  # noqa: E731
            init_params,
            num_warmup,
            rng_key=rng_key,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )
        if rng_key.ndim == 1:
            init_state = mh_init_fn(init_params, rng_key)
        else:
            # XXX it is safe to run mh_init_fn under vmap despite that mh_init_fn changes some
            # nonlocal variables: momentum_generator, wa_update, trajectory_len, max_treedepth,
            # wa_steps because those variables do not depend on traced args: init_params, rng_key.
            init_state = vmap(mh_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run MH from the given :data:`~numpyro.infer.mcmc.MHState` and return the resulting
        :data:`~numpyro.infer.mcmc.MHState`.

        :param MHState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running MH.
        """
        return self._sample_fn(state, model_args, model_kwargs)

    def get_diagnostics_str(self, state):
        return "acc. prob={:.2f}".format(state.mean_accept_prob)
