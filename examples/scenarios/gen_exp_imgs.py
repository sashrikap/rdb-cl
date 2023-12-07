from week8.run_optimal_control import run_opt_week8
from week9.run_optimal_control import run_opt_week9
from week10.run_optimal_control import run_opt_week10

from datetime import datetime
import os

# Temporary file to generate images for report

WEEK_8_EXPS = ["Week8_01", "Week8_02", "Week8_03", "Week8_04"]
WEEK_9_EXPS = ["Week9_01", "Week9_02", "Week9_03", "Week9_04"]
WEEK_10_EXPS = ["Week10_01", "Week10_02"]
ALL_EXPS = WEEK_8_EXPS + WEEK_9_EXPS + WEEK_10_EXPS

week_fns = {
    "Week8": run_opt_week8,
    "Week9": run_opt_week9,
    "Week10": run_opt_week10,
}

env_tasks = {
    "Week8_01": (0, 0),
    "Week8_02": (0, 0),
    "Week8_03": (0, 0),
    "Week8_04": (0, 0),
    "Week9_01": (0, 0, 0.5, 0.5),
    "Week9_02": (0, 0, 0.5, 0.5),
    "Week9_03": (0, 0, 0.5, 0.5),
    "Week9_04": (0, 0, 0.5, 0.5, -0.5, -0.5),
    "Week10_01": (0, 0),
    "Week10_02": (0, 0),
}

for exp_set in [WEEK_8_EXPS, WEEK_9_EXPS, WEEK_10_EXPS]:
    fn = week_fns[exp_set[0].split("_")[0]]
    for exp in exp_set:
        curr_task = "Week8_01"
        folder = f"experiments/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        os.mkdir(folder)
        os.mkdir(f"{folder}/exp_weights")
        os.mkdir(f"{folder}/exp_results")
        fn(exp, env_tasks[exp], folder, 'experiments/standard.json')