from datetime import datetime
import os
import random

from week8.run_optimal_control import run_opt_week8
from week9.run_optimal_control import run_opt_week9
from week10.run_optimal_control import run_opt_week10

"""
Exps:
1. (Week9_04) Cones in both left and right lane 
2. (Week9_03) Fallen tree in the center lane 
3. (Week8_02) Truck in left lane
4. (Week8_04) Three scooters in left lane 
5. (Week8_01) Truck in left and non-ego car in center ahead of ego car 
5. (Week8_03) Motorcycle left and slightly ahead
6. (Week10_01) Two autonomous cars merging into left lane
6. (Week10_02) Two autonomous cars merging into left lane sashrika
where there's a truck 
"""
WEEK_8_EXPS = ["Week8_01", "Week8_02", "Week8_03", "Week8_04"]
WEEK_9_EXPS = ["Week9_01", "Week9_02", "Week9_03", "Week9_04"]
WEEK_10_EXPS = ["Week10_01", "Week10_02"]
ALL_EXPS = WEEK_8_EXPS + WEEK_9_EXPS + WEEK_10_EXPS

week_fns = {
    "Week8": run_opt_week8,
    "Week9": run_opt_week9,
    "Week10": run_opt_week10,
}
week_info = {
    "Week8_01": {
        'num_obstacles': 0,
        'num_vehicles': 2,
        'num_auto_vehicles': 0,
    },
    "Week8_02": {
        'num_obstacles': 0,
        'num_vehicles': 1,
        'num_auto_vehicles': 0,
    },
    "Week8_03": {
        'num_obstacles': 0,
        'num_vehicles': 1,
        'num_auto_vehicles': 0,
    }, 
    "Week8_04": {
        'num_obstacles': 0,
        'num_vehicles': 3,
        'num_auto_vehicles': 0,
    },
    "Week9_01": {
        'num_obstacles': 1,
        'num_vehicles': 0,
        'num_auto_vehicles': 0,
    },
    "Week9_02": {
        'num_obstacles': 1,
        'num_vehicles': 0,
        'num_auto_vehicles': 0,
    },
    "Week9_03": {
        'num_obstacles': 1,
        'num_vehicles': 0,
        'num_auto_vehicles': 0,
    },
    "Week9_04": {
        'num_obstacles': 2,
        'num_vehicles': 0,
        'num_auto_vehicles': 0,
    },
    "Week10_01": {
        'num_obstacles': 0,
        'num_vehicles': 0,
        'num_auto_vehicles': 1,
    },
    "Week10_02": {
        'num_obstacles': 0,
        'num_vehicles': 1,
        'num_auto_vehicles': 1,
    },

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

def order_by_heuristic(heuristic, exps):
    assert heuristic in ["", "num_obstacles", "num_vehicles", "num_auto_vehicles"]
    return exps if heuristic == "" else sorted(exps, key=lambda x: week_info[x][heuristic])

def chain_exps(EXPS, starter_weights_file):
    # initializing starter weights to all be 0.1 
    # except for fence and lane weights (2)
    weights_file = starter_weights_file

    folder = f"experiments/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    os.mkdir(folder)
    os.mkdir(f"{folder}/exp_weights")
    os.mkdir(f"{folder}/exp_results")

    print("running experiments: ", EXPS)
    print("starter: ", starter_weights_file)
    print("saved to folder: ", folder)

    for exp in EXPS:
        run_opt = week_fns[exp.split("_")[0]]
        run_opt(ENV_NAME=exp, TASK=env_tasks[exp], folder=folder, weights_file=weights_file)

        # update subsequent weights_file to be output of prev experiment (i.e. transference)
        weights_file = f"{folder}/exp_weights/{exp}.json"

# for exp in WEEK_9_EXPS + WEEK_10_EXPS:
#     chain_exps([exp for _ in range(10)], "experiments/standard.json")

# for heuristic in ["num_obstacles", "num_vehicles", "num_auto_vehicles"]:
#     print(f"heuristic {heuristic}")
#     exp_lst = order_by_heuristic(heuristic, ALL_EXPS[:])

    # chain_exps(exp_lst, "experiments/standard.json")
    # chain_exps(exp_lst, "experiments/uniform.json")

exp_lst = WEEK_8_EXPS[:3] + WEEK_9_EXPS[:3] + WEEK_10_EXPS + [WEEK_8_EXPS[-1]] + [WEEK_9_EXPS[-1]]
print(exp_lst)
assert len(exp_lst) == 10
chain_exps(exp_lst, "experiments/standard.json")
# 1. Week8_01
# 2. Week8_02
# 3. Week8_03
# 4. Week9_01
# 5. Week9_02
# 6. Week9_03
# 7. Week10_01
# 8. Week10_02
# 9. Week8_04
# 10. Week9_04