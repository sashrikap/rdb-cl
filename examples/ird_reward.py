import gym
import time, copy
import numpy as np
import rdb.envs.drive2d
from tqdm import tqdm
from rdb.optim.open import shooting_optimizer
from rdb.optim.runner import Runner
from rdb.visualize.plot import plot_3d, plot_episode
from rdb.visualize.save import save_rewards
from rdb.visualize.preprocess import normalize_features

## Handles
VIZ_TRAINING = False
FRAME_WIDTH = 450

## Environment setup
env = gym.make("Week3_02-v0")
env.reset()
main_car = env.main_car
udim = 2
horizon = 10
T = 20
optimizer = shooting_optimizer(
    env.dynamics_fn, main_car.cost_runtime, udim, horizon, env.dt, T=T
)
runner = Runner(env, main_car)

## Training environments
weights = {
    "dist_cars": 100.0,
    "dist_lanes": 10.0,
    "dist_fences": 300.0,
    "speed": 20.0,
    "control": 80.0,
}
train_pairs = [(0.4, -0.2), (0.7, 0.2)]

## Visualize Training environments
# Bad: (0.3, -0.1)
for idx, (y0, y1) in enumerate(train_pairs):
    env.set_init_state(y0, y1)
    env.reset()
    state = env.state

    actions = optimizer(state, weights=weights)
    traj, cost, info = runner(state, actions)
    print(f"Training [{idx}]")
    print(f"Total cost: {cost:.3f}")
    text = f"Total cost: {cost:.3f}\nTraining [{idx}]"
    total_cost = 0
    mode = "human" if VIZ_TRAINING else "rgb_array"
    frames = runner.collect_frames(actions, FRAME_WIDTH, mode, text)
    plot_episode(frames, f"Training [{idx}]")


## Testing Environments
y0_range = np.arange(-0.5, 0.51, 0.1)
y1_range = np.arange(-0.5, 0.51, 0.1)
# y0_range = np.arange(-0.5, 0.51, 0.5)
# y1_range = np.arange(-0.5, 0.51, 0.5)
num0, num1 = len(y0_range), len(y1_range)
rews = [[] for _ in range(num0)]
feats_keys = env.get_features_keys()
feats_sum = {key: np.zeros((num0, num1)) for key in feats_keys}

for i0, y0 in enumerate(tqdm(y0_range)):
    for i1, y1 in enumerate(y1_range):
        env.set_init_state(y0, y1)
        env.reset()
        state = env.state

        actions = optimizer(state, weights=weights)
        traj, cost, info = runner(state, actions)
        rews[i0].append(-1.0 * cost)
        # print(f"y0 ({y0:.2f}) y1 ({y1:.2f}) cost ({cost:.2f})")

        for key in info["feats_sum"].keys():
            feats_sum[key][i0][i1] = info["feats_sum"][key]

feats_sum, normalize_info = normalize_features(feats_sum)

# save_rewards(y0_range, y1_range, rews, "data/ird_reward.npz")
plot_3d(
    y0_range,
    y1_range,
    all_rews,
    xlabel="Car 0 Position",
    ylabel="Car 1 Position",
    title="Reward",
)
