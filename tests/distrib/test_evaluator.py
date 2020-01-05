"""Test designer module.
"""
from rdb.infer.ird_oc import IRDOptimalControl, Designer
from rdb.distrib.evaluator import EvaluatorParticle
from rdb.optim.mpc import shooting_method
from rdb.infer.utils import *
from functools import partial
from jax import random
import numpyro.distributions as dist
import gym, rdb.envs.drive2d
import ray

ray.init()

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

NUM_DESIGNERS = 10
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
designer = Designer(
    rng_key=None,
    env=env,
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


# def run_evaluator():
#     key = random.PRNGKey(0)
#     designer.update_key(key)
#     ps = []
#     num_workers = 1
#     ps = designer.sample(task, str(task))
#     evaluator = EvaluatorParticle(env_fn, controller_fn)
#     evaluator.evaluate(key, ps.weights, true_w, task, str(task))

# run_evaluator()


def test_evaluator():
    key = random.PRNGKey(0)
    designer.update_key(key)
    ps = []
    num_workers = 1
    for _ in range(num_workers):
        ps.append(designer.sample(task, str(task)))
    evals = [
        EvaluatorParticle.remote(env_fn, controller_fn) for _ in range(num_workers)
    ]
    [
        e.evaluate.remote(key, p.weights, true_w, task, str(task))
        for e, p in zip(evals, ps)
    ]
    [print(ray.get(e.get_result.remote())) for e in evals]


test_evaluator()


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
