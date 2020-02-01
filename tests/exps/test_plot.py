import numpy as onp
from rdb.visualize.plot import *


def test_perf():
    N = 100
    perf = onp.random.random(N)
    violations = onp.random.random(N)
    task_names = ["" for _ in range(N)]
    path = "./test_plot"
    # plot_rankings(task_names, perf, "perf", [violations], ["violations"], path=path)
