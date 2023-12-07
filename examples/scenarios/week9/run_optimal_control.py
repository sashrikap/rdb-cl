import gym
import jax
import copy
import jax.numpy as jnp
import numpy as onp
import rdb.envs.drive2d
import json

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer import *
from rdb.visualize.render import render_env
from rdb.visualize.preprocess import normalize_features
from rdb.infer.dictlist import *
from PIL import Image

from utils import format_weights_dict

def run_opt_week9(ENV_NAME, TASK, folder, weights_file=None):
    DUMMY_ACTION = False
    DRAW_HEAT = False
    REPLAN = True
    MAKE_MP4 = True
    ENGINE = "scipy"
    METHOD = "lbfgs"

    env = gym.make(ENV_NAME)
    env.reset()
    main_car = env.main_car
    horizon = 10
    T = 30
    fname = f"{ENV_NAME}_01"

    with open(weights_file) as json_file:
        weights = json.load(json_file)

    if TASK == "RANDOM":
        num_tasks = len(env.all_tasks)
        print(f"Total tasks {num_tasks}")
        TASK = env.all_tasks[onp.random.randint(0, num_tasks)]
    env.set_task(TASK)
    env.reset()

    print(f"Task {TASK}")

    if not REPLAN:
        T = horizon
    optimizer, runner = build_mpc(
        env,
        main_car.cost_runtime,
        horizon,
        env.dt,
        replan=REPLAN,
        T=T,
        engine=ENGINE,
        method=METHOD,
    )
    state = copy.deepcopy(env.state)
    t1 = time()
    w_list = DictList([weights])
    w_list = w_list.prepare(env.features_keys)
    actions = optimizer(state, weights=None, weights_arr=w_list.numpy_array())
    traj, cost, info = runner(state, actions, weights=weights, batch=False)
    
    output_dict = {}
    for key, val in info["feats_sum"].items():
        output_dict[key] = val.item()
        print(f"Feature {key}: {val.item()}")

    env.reset()
    env.render("human", draw_heat=DRAW_HEAT, weights=weights)
    frames = []

    for t in range(T):
        env.step(actions[:, t])
        img = env.render("rgb_array")
        frames.append(img)
        sleep(0.2)


    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(f"render_optimal_control_{fname}.gif", save_all = True, append_images=imgs[1:], duration=100, loop=0)

    with open(f'{folder}/exp_weights/{ENV_NAME}.json', 'w') as json_file:
        json.dump(format_weights_dict(output_dict), json_file, indent=2)

    # also save exp results for violations and costs
    output_dict['cost'] = cost.item()
    print("cost: ", cost.item())
    for key, val in info["vios_sum"].items():
        output_dict[key] = val.item()
        
    with open(f'{folder}/exp_results/{ENV_NAME}.json', 'w') as json_file:
        json.dump(output_dict, json_file, indent=2)