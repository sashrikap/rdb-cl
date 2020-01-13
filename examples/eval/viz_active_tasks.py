import os
import jax.numpy as np
import matplotlib.pyplot as plt
import gym, rdb.envs.drive2d
from rdb.optim.mpc import shooting_method


def read_tasks(path):
    # print(path)
    idx = path.index("seed_")
    seed = path.replace(".npz", "")[idx + len("seed_") :]
    tasks = np.load(path, allow_pickle=True)["curr_tasks"].item()
    return seed, tasks


def plot_tasks(path):
    seeds = []
    seed_paths = []
    seed_tasks = []
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    for file in sorted(os.listdir(dirname)):
        if filename in file:
            filepath = os.path.join(dirname, file)
            if os.path.isfile(filepath):
                seed_paths.append(filepath)

    # print(seed_paths)
    for path in seed_paths:
        seed, tasks = read_tasks(path)
        seeds.append(seed)
        seed_tasks.append(tasks)

    os.makedirs(f"{data_path}/tasks", exist_ok=True)
    for seed, tasks in zip(seeds, seed_tasks):
        for method, method_tasks in tasks.items():
            for itr, task in enumerate(method_tasks):
                env.set_task(task)
                env.reset()
                state = env.state
                path = f"{data_path}/tasks/tasks_seeds_{seed}_method_{method}_itr_{itr:02d}.png"
                runner.collect_thumbnail(
                    state, np.zeros((env.udim, HORIZON)), path=path
                )


if __name__ == "__main__":
    data_path = "data/200110/active_ird_exp_mid"
    ENV_NAME = "Week6_01-v0"
    env = gym.make(ENV_NAME)
    env.reset()
    HORIZON = 10
    optimizer, runner = shooting_method(
        env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
    )

    plot_tasks(data_path)
