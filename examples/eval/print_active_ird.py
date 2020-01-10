import os
import jax.numpy as np
from jax import random
from rdb.infer.particles import Particles

SAVEPATH = "data/200107/active_ird_exp_mid/save"

FILEPAIRS = {}

for filename in os.listdir(SAVEPATH):
    fname = filename.replace(".npz", "").split("_")
    itr = int(fname[fname.index("itr") + 1])
    method = fname[fname.index("method") + 1]
    seed = fname[fname.index("seed") + 1]
    pair = (itr, os.path.join(SAVEPATH, filename))

    if method not in FILEPAIRS.keys():
        FILEPAIRS[method] = {}
    if seed not in FILEPAIRS[method].keys():
        FILEPAIRS[method][seed] = []
    FILEPAIRS[method][seed].append(pair)

"""
FILEPAIRS:
  - key: method
    - key: seed
      - list: filenames
"""
# print(FILEPAIRS)
for method in FILEPAIRS.keys():
    for seed in FILEPAIRS[method].keys():
        list_ = FILEPAIRS[method][seed]
        FILEPAIRS[method][seed] = [x[1] for x in sorted(list_, key=lambda x: x[0])]

key = random.PRNGKey(0)
env, controller, runner, sample_ws = None, None, None, None


def env_fn():
    return None


ps = Particles(key, env_fn, controller, runner, sample_ws)

method = "ratiomean"
seed = "[0 9]"

print(f"IRD Belief")
print(f"Method {method} seed {seed}")
for itr, filename in enumerate(FILEPAIRS[method][seed]):
    print(filename)
    ps.load(filename)
    import pdb

    pdb.set_trace()
    ps.log_samples(4)

print(f"\nIRD Observation")
print(f"Method {method} seed {seed}")
SAVEPATH = "data/191231_test/active_ird_exp1"
obs_path = f"{SAVEPATH}_seed_{seed}.npz"
obs_data = np.load(obs_path, allow_pickle=True)
print(f"Obs {obs_data['curr_obs']}")
