"""Informed Designer Experiment.

See how we can construct the best prior knowledge for reward designer.

Full Informed Designer Experiment:
    1)

"""

from rdb.infer.utils import random_choice
from rdb.infer.particles import Particles
from rdb.exps.utils import Profiler
from numpyro.handlers import seed
from tqdm.auto import tqdm
from time import time
import jax.numpy as np
import numpy as onp
import copy
import os


class ExperimentDesignerPrior(object):
    """Informed Designer Experiment.

    Args:
        designer (object): PGM-based reward designer
        num_prior (int): number of latent tasks

    """

    def __init__(self, designer, eval_server, num_prior):
        self._designer = designer
        self._eval_server = eval_server
        self._num_prior = num_prior

    def run(self, task):
        """Simulate designer on `task`. Varying the number of latent
        tasks as prior
        """
        for n_latent in range(self._num_prior):
            print(f"Latent {n_latent}")
