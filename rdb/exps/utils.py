from os.path import join, expanduser, isdir, dirname
from itertools import product
from jax import random
import matplotlib.pyplot as plt
import jax.numpy as np
import logging, time
import numpy as onp
import pathlib, os
import copy
import yaml
import rdb

## ===============================================
## ============== Save & Load Tools ==============
## ===============================================


PROJECT_DIR = dirname(rdb.__path__[0])


def examples_dir():
    return join(PROJECT_DIR, "examples")


def data_dir():
    return join(PROJECT_DIR, "data")


def str_to_key(seed_str):
    assert seed_str[0] == "[" and seed_str[-1] == "]"
    seed = seed_str.replace("]", "")
    seed = seed.replace("[", "")
    seed = seed.strip()
    assert (
        int(seed.split()[0]) == 0 and len(seed.split()) == 2
    ), f"Invalid seed str {seed_str}"
    rng_key = random.PRNGKey(int(seed.split()[1]))
    return rng_key


def load_params(filepath):
    """Load experiment parameters from yaml"""
    with open(filepath, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def save_params(filepath, params):
    """Save experiment parameters to yaml"""
    with open(filepath, "w+") as stream:
        try:
            yaml.dump(params, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)


def create_params(template, params):
    """Return products of all params specified as list.

    Example:
        >> template = {'a': [1], 'b': [2]}
        >> create_params(template, {
            'a': [1, 2],
            'b': [2, 3]
           })
        >> [{'a': [1], 'b': [2]}, {'a': [1], 'b': [3]}, {'a': [2], 'b': [2]}, {'a': [2], 'b': [3]}]

    """
    all_params = []
    list_keys, list_vals = [], []
    for key, val in params.items():
        if type(val) == list and len(val) > 0:
            list_keys.append(key)
            list_vals.append(val)
        else:
            template[key] = val
    if len(list_keys) == 0:
        return [template]
    else:
        prod_vals = product(*list_vals)
        for prod in prod_vals:
            tparam = copy.deepcopy(template)
            for key, val in zip(list_keys, prod):
                tparam[key] = [val]
            all_params.append(tparam)
        return all_params


## ===============================================
## ============== Experiment Tools ===============
## ===============================================


def normalize_weights(weights, key=None):
    assert key is not None, "Normalized key not provided"
    assert key in weights, "Normalized key not included in weights"
    eps = 1e-8
    output = copy.deepcopy(weights)
    factor = output[key] + eps
    for k, v in output.items():
        output[k] = round(v / factor, 5)
    return output


## ===============================================
## ================= Speed Tools =================
## ===============================================


class Profiler(object):
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def step(self, name):
        """ Returns the duration and stepname since last step/start """
        self.summarize_step(start=self.step_start, step_name=name)
        now = time.time()
        self.step_start = now

    def __enter__(self):
        self.start = time.time()
        self.step_start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.summarize_step(self.start)

    def summarize_step(self, start, step_name=""):
        duration = time.time() - start
        step_semicolon = ":" if step_name else ""
        if self.verbose:
            print(
                f"{self.name}{step_semicolon + step_name}: {duration:.3f} seconds {1/(duration + 1e-3):.3f} fps"
            )
        return duration
