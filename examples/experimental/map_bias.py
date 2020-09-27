import jax.numpy as jnp
import numpy as onp
from jax import random
from tqdm import trange
from rdb.infer import *

rng_key = random.PRNGKey(0)


def test_map(num_runs=10):
    global rng_key
    map_means = []
    unif_means = []
    for ri in trange(num_runs):
        rand_weights = []
        N = 10000
        N_map = int(N / 100.0)
        max_weights = 10
        N_bins = 200
        weight_params = {"max_weights": max_weights, "bins": N_bins}
        for _ in range(N):
            val = max_weights * (onp.random.uniform() - 0.5) * 2
            rand_weights.append({"normalized": 1.0, "test_map": val})
        rand_weights = DictList(rand_weights)

        ps = Particles(
            rng_name="test_map",
            rng_key=None,
            env_fn=None,
            controller=None,
            runner=None,
            save_name="test_map",
            weights=rand_weights,
            normalized_key="normalized",
            weight_params=weight_params,
            save_dir="data/test",
            fig_dir="data/test",
        )
        # ps.visualize(log_scale=True)
        ws_map = ps.map_estimate(N_map, log_scale=True).weights
        map_means.append(ws_map["test_map"].mean())
        # import pdb; pdb.set_trace()
        rng_key, sample_key = random.split(rng_key)
        ps.update_key(sample_key)
        ws_unif = ps.subsample(N_map).weights
        unif_means.append(ws_unif["test_map"].mean())
    map_means = jnp.array(map_means)
    unif_means = jnp.array(unif_means)
    print(f"Map means {map_means.mean():.3f} map std {map_means.std():.3f}")
    print(f"Uniform means {unif_means.mean():.3f} map std {unif_means.std():.3f}")


if __name__ == "__main__":
    test_map(num_runs=40)
