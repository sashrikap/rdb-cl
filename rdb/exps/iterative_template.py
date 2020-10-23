# `nb_templater` generated Python script
# Generated from .ipynb template: ../../../../data/200501/iterative_divide/notebook_rng_00.ipynb
# www.github.com/ismailuddin/jupyter-nb-templater/
# Generated on: 2020-05-03 20:30

from rdb.exps.utils import *
import nbformat as nbf
import sys, os
import shutil
import rdb

nb = nbf.v4.new_notebook()
data_root = os.path.join(os.path.dirname(rdb.__path__[0]), "data")


def build_cells(active_fns, n_iteration, rng_key_val, exp_name):
    cell_text_0 = """\
# User study notebook

Thank you for taking part in this study. In this task you will be designing reward functions for autonomous cars to perform lane change in face of other vehicles and obstacles.

"""

    cell_text_1 = f"""\
## Start experiment (run following block)"""

    cell_2 = """\
import gym
import copy
import rdb.envs.drive2d
from jax import random
from rdb.infer.utils import *
from functools import partial
from rdb.optim.mpc import build_mpc
from rdb.exps.iterative_ird import run_iterative
%load_ext autoreload
%autoreload 2"""

    cell_3 = """\
%%html
<style>
.reminder {
    pointer-events: none;
}
</style>"""

    cell_text_4 = """\
## Prepare experiment (run this block)"""

    cell_5 = f"""\
experiment = run_iterative(evaluate=False, gcp_mode=False, use_local_params=True)
experiment.set_exp_name('{exp_name}')
rng_key_val = {rng_key_val}
rng_key = random.PRNGKey(rng_key_val)
method_sequences = dict()

!mkdir -p yaml
!mkdir -p mp4
!mkdir -p img

experiment.update_key(rng_key_val)
experiment.reset()
active_keys = list(experiment._active_fns.keys())
tasks = dict.fromkeys(experiment._active_fns.keys(), [])"""

    def make_iteration_cells(iteration):
        iter_cell_text_1 = f"""\
## Round {iteration} (run this block)"""

        iter_cell_2 = f"""\
iteration={iteration}
rng_random, rng_key = random.split(rng_key)
n_active = len(active_keys)
rand_ind = random_choice(rng_random, onp.arange(n_active), shape=(n_active,), replace=False)
method_sequences[iteration] = [active_keys[i] for i in rand_ind]"""

        def make_method_cells(method_idx):
            if iteration == 0:
                method_cell_text_1 = f"""\
## Round {iteration} (run this block and use the widget to enter design)"""
                method_cell_2 = f"""\
experiment.query_user_input_from_notebook(method_sequences[{iteration}])"""
            else:
                method_cell_text_1 = f"""\
## Round {iteration} method {method_idx} (run this block and use the widget to enter design)"""

                method_cell_2 = f"""\
method = method_sequences[{iteration}][{method_idx}]
experiment.query_user_input_from_notebook(method)"""

            return [
                nbf.v4.new_markdown_cell(method_cell_text_1),
                nbf.v4.new_code_cell(method_cell_2),
            ]

        iter_cell_text_4 = f"""\
## Round {iteration} Propose (run this block and wait)"""

        iter_cell_5 = """\
experiment._save()
experiment.propose()"""

        iter_cell_text_6 = f"""\
## Round {iteration} Survey (run this block and enter the scores)"""

        iter_cell_9 = """\
experiment._save()
experiment.show_proposed_tasks_for_feedback()"""

        pre_cells = [
            nbf.v4.new_markdown_cell(iter_cell_text_1),
            nbf.v4.new_code_cell(iter_cell_2),
        ]
        method_cells = []
        if iteration == 0:
            method_cells += make_method_cells(0)
        else:
            for im, method in enumerate(active_fns):
                method_cells += make_method_cells(im)
        post_cells = [
            nbf.v4.new_markdown_cell(iter_cell_text_4),
            nbf.v4.new_code_cell(iter_cell_5),
            nbf.v4.new_markdown_cell(iter_cell_text_6),
            nbf.v4.new_code_cell(iter_cell_9),
        ]

        return pre_cells + method_cells + post_cells

    cell_14 = """\

"""

    nb["cells"] = [
        nbf.v4.new_markdown_cell(cell_text_0),
        nbf.v4.new_markdown_cell(cell_text_1),
        nbf.v4.new_code_cell(cell_2),
        nbf.v4.new_code_cell(cell_3),
        nbf.v4.new_markdown_cell(cell_text_4),
        nbf.v4.new_code_cell(cell_5),
    ]

    for iteration in range(n_iteration):
        nb["cells"] += make_iteration_cells(iteration)

    nb["cells"] += [nbf.v4.new_code_cell(cell_14)]
    os.makedirs(save_dir, exist_ok=True)
    nbf.write(nb, f"{save_dir}/{exp_name}_rng_{rng_key_val:02d}.ipynb")
    print(
        f"Jupyter notebook {save_dir}/{exp_name}_{rng_key_val:02d}.ipynb successfully generated."
    )


if __name__ == "__main__":
    rng_keys = list(range(4))
    copy_params = True
    n_iteration = 4
    # divide_conquer = False
    # active_fns = ["random", "infogain", "ratiomean", "ratiomin", "difficult"]
    divide_conquer = True
    active_fns = ["random", "infogain", "difficult"]
    data_dir = "201012"
    exp_name = f"iterative_divide_initial_1v1_icra_20"
    # exp_name = f"iterative_divide_test"
    save_dir = f"{data_root}/{data_dir}/{exp_name}/"

    for rng_key_val in rng_keys:
        build_cells(active_fns, n_iteration, rng_key_val, exp_name)
    if copy_params:
        params_path = f"{examples_dir()}/params/iterative_template.yaml"
        exp_params_path = f"{save_dir}/iterative_template.yaml"
        shutil.copy(params_path, exp_params_path)
