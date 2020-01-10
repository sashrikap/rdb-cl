import gym
import rdb.envs.drive2d
import numpyro.distributions as dist
from jax import random
from numpyro.handlers import seed
from rdb.infer.utils import random_choice, prior_sample, prior_log_prob

key = random.PRNGKey(1)
env = gym.make("Week3_02-v0")
env.update_key(key)
# print(env.sample_tasks(3))

arr = [1, 2, 3, 4, 5]
random_choice = seed(random_choice, rng_seed=key)
prior_sample = seed(prior_sample, rng_seed=key)

log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.01),
    "dist_lanes": dist.Uniform(-10, 10),
}

print(random_choice(arr, 3))
sample = prior_sample(log_prior_dict)
print(sample)
print(prior_log_prob(sample, log_prior_dict))

env.update_key(key)
print(env.sample_tasks(3))
