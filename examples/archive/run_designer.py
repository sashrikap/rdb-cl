"""Run Informed-Designer Experiment Loop.

Note:
    * See (rdb.exps.designer_prior.py) for more details

"""

from rdb.exps.designer_prior import ExperimentDesignerPrior
from rdb.distrib.particles import ParticleServer
from rdb.infer.particles import Particles
from rdb.infer.ird_oc import Designer
from rdb.optim.mpc import build_mpc
from rdb.exps.utils import *
from functools import partial
from rdb.infer import *
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
        controller, runner = build_mpc(
            env, env.main_car.cost_runtime, HORIZON, env.dt, replan=False
        )
        return controller, runner

    ## Prior sampling & likelihood functions for PGM
    prior = LogUniformPrior(
        rng_key=None,
        normalized_key=NORMALIZED_KEY,
        feature_keys=FEATURE_KEYS,
        log_max=MAX_WEIGHT,
    )
    proposal = IndGaussianProposal(
        rng_key=None,
        normalized_key=NORMALIZED_KEY,
        feature_keys=FEATURE_KEYS,
        proposal_var=DESIGNER_PROPOSAL_VAR,
    )

    ## Evaluation Server
    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=NUM_EVAL_WORKERS, parallel=PARALLEL
    )

    _env = env_fn()
    _controller, _runner = controller_fn(_env)
    weight_params = {"bins": HIST_BINS, "max_weights": MAX_WEIGHT}
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"

    designer = Designer(
        rng_key=None,
        env_fn=env_fn,
        controller=_controller,
        runner=_runner,
        beta=BETA,
        true_w=TRUE_W,
        prior=prior,
        proposal=proposal,
        normalized_key=NORMALIZED_KEY,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
        weight_params=weight_params,
        sampler_args={
            "num_warmups": NUM_WARMUPS,
            "num_samples": NUM_DESIGNERS,
            "num_chains": NUM_DESIGNER_CHAINS,
        },
        use_true_w=False,
    )
    if FIXED_TASK_SEED is not None:
        fixed_task_seed = random.PRNGKey(FIXED_TASK_SEED)
    else:
        fixed_task_seed = None

    experiment = ExperimentDesignerPrior(
        rng_key=None,
        designer=designer,
        eval_server=eval_server,
        num_eval_tasks=NUM_EVAL_TASKS,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=EXP_NAME,
        fixed_task_seed=fixed_task_seed,
    )
    for ki in RANDOM_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        for itr in range(NUM_ITERATIONS):
            experiment.run(TASK, num_prior_tasks=itr, evaluate=EVALUATE)


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
