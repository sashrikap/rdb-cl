"""Run Informed-Designer Experiment Loop.

Note:
    * See (rdb.exps.designer_prior.py) for more details

"""

from rdb.exps.designer_prior import ExperimentDesignerPrior
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import Designer
from rdb.optim.mpc import shooting_method
from rdb.infer.particles import Particles
from rdb.exps.utils import load_params
from functools import partial
from rdb.infer.utils import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse
import argparse


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
        "dist_cars": dist.Uniform(-0.01, 0.01),
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
    prior_log_prob_fn = build_log_prob_fn(log_prior_dict)
    prior_sample_fn = build_prior_sample_fn(log_prior_dict)
    proposal_fn = build_gaussian_proposal(proposal_std_dict)

    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    )

    _env = env_fn()
    _controller, _runner = controller_fn(_env)
    _truth = Particles(None, env_fn, _controller, _runner, [TRUE_W])

    designer = Designer(
        rng_key=None,
        env_fn=env_fn,
        controller=_controller,
        runner=_runner,
        beta=BETA,
        truth=_truth,
        prior_log_prob_fn=prior_log_prob_fn,
        proposal_fn=proposal_fn,
        sampler_args={"num_warmups": NUM_WARMUPS, "num_samples": NUM_DESIGNERS},
        use_true_w=False,
    )

    SAVE_ROOT = "data" if not GCP_MODE else "/gcp_output"
    experiment = ExperimentDesignerPrior(
        rng_key=None,
        designer=designer,
        eval_server=eval_server,
        num_eval_tasks=NUM_EVAL_TASKS,
        save_dir=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=EXP_NAME,
    )
    for ki in RANDOM_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        for itr in range(NUM_ITERATIONS):
            experiment.run(TASK, num_prior_tasks=itr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE

    # Load parameters
    if not GCP_MODE:
        params = load_params("examples/params/designer_template.yaml")
    else:
        params = load_params("/dar_payload/rdb/examples/params/designer_params.yaml")
    locals().update(params)
    main()
