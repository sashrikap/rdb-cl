"""Test designer module.
"""
from rdb.infer.ird_oc import IRDOptimalControl, Designer
from rdb.distrib.particles import ParticleServer
from rdb.optim.mpc import shooting_method
from rdb.infer.utils import *
from functools import partial
from jax import random
from time import time
import numpyro.distributions as dist
import gym, rdb.envs.drive2d
import ray

ENV_NAME = "Week6_01-v0"
NUM_WARMUPS = 100
NUM_ACTIVE_TASKS = 16
USER_TRUE_W = False

## Faster sampling
NUM_NORMALIZERS = 200
NUM_SAMPLES = 500
NUM_ACTIVE_SAMPLES = -1
NUM_EVAL_SAMPLES = -1
NUM_EVAL_TASKS = 8

NUM_DESIGNERS = 40
MAX_WEIGHT = 8.0
BETA = 1.0
HORIZON = 10
EXP_ITERATIONS = 8
PROPOSAL_VAR = 0.2

env = gym.make(ENV_NAME)
env.reset()
controller, runner = shooting_method(
    env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
)
true_w = {
    "dist_cars": 1.0,
    "dist_lanes": 0.1,
    "dist_fences": 0.35,
    "dist_objects": 1.25,
    "speed": 0.05,
    "control": 0.1,
}

task = (0.2, -0.7, 0.0, 0.4)
log_prior_dict = {
    "dist_cars": dist.Uniform(0.0, 0.01),
    "dist_lanes": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "dist_fences": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "dist_objects": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "speed": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
    "control": dist.Uniform(-MAX_WEIGHT, MAX_WEIGHT),
}
proposal_std_dict = {
    "dist_cars": 1e-6,
    "dist_lanes": PROPOSAL_VAR,
    "dist_fences": PROPOSAL_VAR,
    "dist_objects": PROPOSAL_VAR,
    "speed": PROPOSAL_VAR,
    "control": PROPOSAL_VAR,
}
prior_log_prob_fn = partial(prior_log_prob, log_prior_dict=log_prior_dict)
proposal_fn = partial(gaussian_proposal, log_std_dict=proposal_std_dict)


def env_fn():
    import gym, rdb.envs.drive2d

    env = gym.make(ENV_NAME)
    env.reset()
    return env


def controller_fn(env_):
    controller, runner = shooting_method(
        env_, env_.main_car.cost_runtime, HORIZON, env_.dt, replan=False
    )
    return controller, runner


designer = Designer(
    rng_key=None,
    env_fn=env_fn,
    controller=controller,
    runner=runner,
    beta=BETA,
    true_w=true_w,
    prior_log_prob=prior_log_prob_fn,
    proposal_fn=proposal_fn,
    sample_method="mh",
    sampler_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_DESIGNERS},
    use_true_w=USER_TRUE_W,
)

key = random.PRNGKey(0)
designer.update_key(key)
num_tasks = 36


def test_single_evaluator():
    particles = designer.sample(task, str(task))
    num_workers = 1
    server = ParticleServer(
        env_fn,
        controller_fn,
        num_workers=num_workers,
        parallel=False,
        initialize_wait=True,
    )
    t1 = time()
    particles = server.compute_tasks(
        particles, [task] * num_tasks, [str(task)] * num_tasks, verbose=True
    )
    print(f"Single worker time {time() - t1:.3f}")


test_single_evaluator()


def test_parallel_evaluator():
    ps = []
    particles = designer.sample(task, str(task))
    num_workers = 4
    server = ParticleServer(
        env_fn, controller_fn, num_workers=num_workers, initialize_wait=True
    )
    t1 = time()
    particles = server.compute_tasks(
        particles, [task] * num_tasks, [str(task)] * num_tasks, verbose=True
    )
    print(f"Parallel worker time {time() - t1:.3f}")


test_parallel_evaluator()


def ttest_ray1():
    import time

    @ray.remote
    class Counter(object):
        def __init__(self):
            self.counter = 0

        def inc(self, val):
            self.counter += val

        def get_counter(self):
            return self.counter

    @ray.remote
    def f(counter):
        for _ in range(1000):
            time.sleep(0.1)
            counter.inc.remote(2)

    counter = Counter.remote()
    # Start some tasks that use the actor.
    [f.remote(counter) for _ in range(3)]
    # f.remote(counter)
    # Print the counter value.
    for _ in range(10):
        time.sleep(1)
        print(ray.get(counter.get_counter.remote()))


def ttest_ray2():
    import time

    @ray.remote
    class Counter(object):
        def __init__(self):
            self.counter = 0

        def inc(self, val):
            time.sleep(3)
            self.counter += val

        def get_counter(self):
            return self.counter

    cs = [Counter.remote() for _ in range(3)]
    [print(ray.get(c.get_counter.remote())) for c in cs]
    [c.inc.remote(2) for c in cs]
    [print(ray.get(c.get_counter.remote())) for c in cs]
    [c.inc.remote(2) for c in cs]
    [print(ray.get(c.get_counter.remote())) for c in cs]
