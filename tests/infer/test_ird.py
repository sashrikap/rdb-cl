from rdb.infer.ird_oc import *
from rdb.infer.algos import *
import jax.numpy as np
import numpyro
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
    cost_runtime = env.main_car.cost_runtime
    horizon = 10
    controller = shooting_optimizer(
        env.dynamics_fn, cost_runtime, env.udim, horizon, env.dt
    )
    runner = Runner(env, cost_runtime=cost_runtime)
    beta = 5.0
    env.reset()
    state = copy.deepcopy(env.state)

    def prior():
        w_dist_cars = 1.0
        w_dist_lanes = numpyro.sample("dist_lanes", dist.Uniform(0, 10))
        w_dist_fences = numpyro.sample("dist_fences", dist.Uniform(0, 10))
        w_speed = numpyro.sample("speed", dist.Uniform(0, 10))
        w_control = numpyro.sample("control", dist.Uniform(0, 10))
        return {
            "dist_cars": np.exp(w_dist_cars),
            "dist_fences": np.exp(w_dist_fences),
            "dist_fences": np.exp(w_dist_fences),
            "speed": np.exp(w_speed),
            "control": np.exp(w_control),
        }

    pgm = IRDOptimalControl(
        env,
        controller,
        runner,
        beta,
        prior_fn=prior,
        method="NUTS",
        sampler_args={"num_samples": 20, "num_warmups": 20},
    )
    user_weights = {
        "dist_cars": 50,
        "dist_lanes": 30.0,
        "dist_fences": 5000.0,
        "speed": 1000.0,
        "control": 20.0,
    }
    _, samples = pgm.posterior(user_weights, state)
    # import pdb; pdb.set_trace()
