import matplotlib.pyplot as plt
import jax.numpy as np
import logging, time
import copy
import yaml
from jax import random
from itertools import product


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
    with open(filepath, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def save_params(filepath, params):
    with open(filepath, "w+") as stream:
        try:
            yaml.dump(params, stream, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)


def create_params(template, params):
    """Return products of all params specified as list.

    Example:
        >> template = {'a': 1, 'b': 2}
        >> create_params(template, {'a': [1, 2], 'b':[2, 3]})
        >> # [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'a': 2, 'b': 2}, {'a': 2, 'b': 3}]

    """
    all_params = []
    list_keys, list_vals = [], []
    for key, val in params.items():
        if type(val) == list and len(val) > 0:
            list_keys.append(key)
            list_vals.append(val)
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


def plot_weights(
    weights_dicts, highlight_dicts=[], highlight_colors=[], path=None, title=None
):
    PLOT_BINS = 100
    MAX_WEIGHT = 8.0

    fig = plt.figure(figsize=(8, 6), dpi=80)
    n_values = len(weights_dicts[0].values())
    for i, key in enumerate(weights_dicts[0].keys()):
        values = [np.log(s[key]) for s in weights_dicts]
        plt.subplot(n_values, 1, i + 1)
        n, bins, patches = plt.hist(
            values,
            PLOT_BINS,
            range=(-MAX_WEIGHT, MAX_WEIGHT),
            density=True,
            facecolor="b",
            alpha=0.75,
        )
        ## Highlight value
        for d, c in zip(highlight_dicts, highlight_colors):
            if d is None:
                continue
            val = d[key]
            plt.axvline(x=np.log(val), c=c)
        plt.title(key)
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()


class Profiler(object):
    def __init__(self, name, level=logging.INFO):
        self.name = name
        self.level = level

    def step(self, name):
        """ Returns the duration and stepname since last step/start """
        self.summarize_step(start=self.step_start, step_name=name, level=self.level)
        now = time.time()
        self.step_start = now

    def __enter__(self):
        self.start = time.time()
        self.step_start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.summarize_step(self.start)

    def summarize_step(self, start, step_name="", level=None):
        duration = time.time() - start
        step_semicolon = ":" if step_name else ""
        # level = level or self.level
        print(
            f"{self.name}{step_semicolon + step_name}: {duration:.3f} seconds {1/duration:.3f} fps"
        )
        return duration
