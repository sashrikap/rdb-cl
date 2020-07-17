"""Experiment to see the effect different manipulations on MCMC prior.
"""

import os
import numpyro
import pathlib
from jax import random
from numpyro import handlers
from rdb.infer.prior import LogUniformPrior
from rdb.visualize.plot import plot_weights, plot_rankings


def test_fixed_key_prior():
    """Hypothesis: fixing 1 key as 1 during sampling is equal to normalizing against
    that key after sampling
    """
    prior_contain = LogUniformPrior("a", ["a", "b", "c"], log_max=5)
    prior_exclude = LogUniformPrior("-", ["a", "b", "c"], log_max=5)

    num_samples = 10000

    with handlers.seed(rng_seed=0):
        # samples_con = numpyro.sample("Contains", prior_contain, obs=None, sample_shape=(num_samples,))
        # samples_exc = numpyro.sample("Excludes", prior_exclude, obs=None, sample_shape=(num_samples,))
        samples_con = prior_contain(num_samples)
        samples_exc = prior_exclude(num_samples)

    for key in samples_exc.keys():
        if key != "a":
            samples_exc[key] = samples_exc[key] / samples_exc["a"]
    samples_exc["a"] = samples_exc["a"] / samples_exc["a"]

    for samples, name in zip(
        [samples_con, samples_exc], ["fig_contains", "fig_excludes"]
    ):
        savepath = os.path.join(pathlib.Path(__file__).parent.absolute(), name)
        plot_weights(
            samples,
            path=savepath,
            title="Prior Sample test",
            max_weights=15,
            bins=100,
            log_scale=False,
        )


if __name__ == "__main__":
    test_fixed_key_prior()
