from rdb.infer.ird_oc import *
from rdb.infer.algos import *
import jax.numpy as np
import numpyro
from jax import random, vmap
import numpyro.distributions as dist


def ttest_PGM_kernel():
    prior_fn = lambda: numpyro.sample("s", dist.Uniform(0.0, 10.0))

    def likelihood(prior, obs, std):
        z = numpyro.sample("z", dist.Normal(prior, std))
        beta = 5.0
        diff = np.abs(z - obs)
        return -beta * diff

    pgm = PGM(prior_fn, likelihood)

    infer = NUTSMonteCarlo(pgm.kernel, 100, 100)
    _, samples = infer.posterior(obs=1.0, std=0.1)
    # import pdb; pdb.set_trace()


def ttest_PGM_method():
    def prior():
        return numpyro.sample("s", dist.Uniform(0.0, 10.0))

    def likelihood(prior, obs, std):
        z = numpyro.sample("z", dist.Normal(prior, std))
        beta = 5.0
        diff = np.abs(z - obs)
        return -beta * diff

    sampler_args = {"num_samples": 100, "num_warmups": 100}
    pgm = PGM(prior, likelihood, method="NUTS", sampler_args=sampler_args)
    _, samples = pgm.posterior(obs=1.0, std=0.1)
    # import pdb; pdb.set_trace()


def test_IRD_OC():
    """Test Optimal Controller """
    import gym
    import time, copy
    import jax.numpy as np
    import rdb.envs.drive2d

    from rdb.optim.mpc import shooting_optimizer
    from rdb.optim.runner import Runner
    from rdb.visualize.render import render_env
    from rdb.visualize.preprocess import normalize_features

    env = gym.make("Week3_02-v0")
    env.reset()
    cost_runtime = env.main_car.cost_runtime
    horizon = 10
    controller = shooting_optimizer(
        env.dynamics_fn, cost_runtime, env.udim, horizon, env.dt
    )
    runner = Runner(env, cost_runtime=cost_runtime)
    beta = 5.0
    env.reset()
    state = copy.deepcopy(env.state)

    def prior_log_prob(state):
        w_log_dist_cars = 0.0
        log_dist_lanes = np.log(state["dist_lanes"])
        if log_dist_lanes < 0 or log_dist_lanes > 10:
            return -np.inf
        log_dist_fences = np.log(state["dist_fences"])
        if log_dist_fences < 0 or log_dist_fences > 10:
            return -np.inf
        log_speed = np.log(state["speed"])
        if log_speed < 0 or log_speed > 10:
            return -np.inf
        log_control = np.log(state["control"])
        if log_control < 0 or log_control > 10:
            return -np.inf
        return 0.0
        """return {
            "dist_cars": np.exp(w_log_dist_cars),
            "dist_fences": np.exp(w_log_dist_fences),
            "dist_fences": np.exp(w_log_dist_fences),
            "speed": np.exp(w_log_speed),
            "control": np.exp(w_log_control),
        }"""

    def proposal(weight):
        std_dict = {"dist_lanes": 1.0, "dist_fences": 1.0, "speed": 1.0, "control": 1.0}
        next_weight = copy.deepcopy(weight)
        for key, val in next_weight.items():
            log_val = np.log(val)
            if key in std_dict.keys():
                std = std_dict[key]
                next_log_val = numpyro.sample("next_weight", dist.Normal(log_val, std))
                next_weight[key] = np.exp(next_log_val)
        return next_weight

    key = random.PRNGKey(1)
    pgm = IRDOptimalControl(
        key, env, controller, runner, beta, prior_log_prob=prior_log_prob
    )
    user_weights = {
        "dist_cars": 50,
        "dist_lanes": 30.0,
        "dist_fences": 5000.0,
        "speed": 1000.0,
        "control": 20.0,
    }

    sampler = MetropolisHasting(
        key, pgm, num_warmups=10, num_samples=10, proposal_fn=proposal
    )
    sampler.init(user_weights)
    samples = sampler.sample(user_weights, init_state=state)
    import pdb

    pdb.set_trace()
