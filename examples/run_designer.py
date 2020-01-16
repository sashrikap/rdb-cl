"""Run Informed-Designer Experiment Loop.

Note:
    * See (rdb.exps.designer_prior.py) for more details

"""

from rdb.exps.designer_prior import ExperimentDesignerPrior
from rdb.optim.mpc import shooting_method
from rdb.infer.ird_oc import DesignerInformed
from rdb.distrib.particles import ParticleServer
from rdb.exps.utils import load_params
from functools import partial
from jax import random
import numpyro.distributions as dist
import yaml, argparse


def main():
    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(ENV_NAME)
        env.reset()
        return env

    def controller_fn(env):
        controller, runner = shooting_method(
            env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
        )
        return controller, runner

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
    prior_sample_fn = partial(prior_sample, log_prior_dict=log_prior_dict)
    norm_sample_fn = partial(
        normalizer_sample, sample_fn=prior_sample_fn, num=NUM_NORMALIZERS
    )
    proposal_fn = partial(gaussian_proposal, log_std_dict=proposal_std_dict)
    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    )

    designer = DesignerInformed(
        rng_key=None,
        env_fn=self._env_fn,
        controller=self._controller,
        runner=self._runner,
        betw=beta,
        true_w=true_w,
        prior_log_prob_fn=prior_log_prob_fn,
        proposal_fn=proposal_fn,
        sample_method=sample_method,
        sampler_args=designer_args,
        use_true_w=use_true_w,
    )

    experiment = ExperimentDesignerPrior(designer, eval_server=eval_server)


if __name__ == "__main__":
    main()
