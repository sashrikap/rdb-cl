"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
import numpyro

numpyro.set_host_device_count(3)

from rdb.exps.task_beta import ExperimentTaskBeta
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc_risk import build_risk_averse_mpc
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import gym, rdb.envs.drive2d
import numpyro.distributions as dist
import yaml, argparse, os
import numpyro
import copy
import ray


# numpyro.enable_validation()


def main(random_key):
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"  # Don't change this line
    DEBUG_ROOT = data_dir() if not GCP_MODE else "/gcp_input"

    # Define random key
    rng_key = random.PRNGKey(random_key)

    def env_fn(env_name=None):
        import gym, rdb.envs.drive2d

        if env_name is None:
            env_name = ENV_NAME
        env = gym.make(env_name)
        env.reset()
        return env

    def designer_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            name=name,
            **DESIGNER_CONTROLLER_ARGS,
        )
        return controller, runner

    def risk_controller_fn(env, name=""):
        if EXP_ARGS["risk_averse"]:
            controller, runner = build_risk_averse_mpc(
                env,
                env.main_car.cost_runtime,
                dt=env.dt,
                name=name,
                **IRD_CONTROLLER_ARGS,
                cost_args={"mode": "trajwise"},
            )
        else:
            controller, runner = build_mpc(
                env,
                env.main_car.cost_runtime,
                dt=env.dt,
                name=name,
                **IRD_CONTROLLER_ARGS,
            )
        return controller, runner

    def ird_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env, env.main_car.cost_runtime, dt=env.dt, name=name, **IRD_CONTROLLER_ARGS
        )
        return controller, runner

    eval_server = ParticleServer(
        env_fn,
        risk_controller_fn,
        parallel=EVAL_ARGS["parallel"],
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        max_batch=EVAL_ARGS["max_batch"],
    )
    eval_server.register("Evaluation", EVAL_ARGS["num_eval_workers"])
    ## Prior sampling & likelihood functions for PGM
    def prior_fn(name="", feature_keys=WEIGHT_PARAMS["feature_keys"]):
        return LogUniformPrior(
            normalized_key=WEIGHT_PARAMS["normalized_key"],
            feature_keys=feature_keys,
            log_max=WEIGHT_PARAMS["max_weights"],
            name=name,
        )

    def designer_fn():
        designer = Designer(
            env_fn=env_fn,
            controller_fn=designer_controller_fn,
            prior_fn=prior_fn,
            weight_params=WEIGHT_PARAMS,
            normalized_key=WEIGHT_PARAMS["normalized_key"],
            save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
            exp_name=EXP_NAME,
            **DESIGNER_ARGS,
        )
        return designer

    designer = designer_fn()
    ird_model = IRDOptimalControl(
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=ird_controller_fn,
        designer=designer,
        prior_fn=prior_fn,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=EXP_NAME,
        **IRD_ARGS,
    )

    experiment = ExperimentTaskBeta(
        ird_model,
        env_fn=env_fn,
        designer_fn=designer_fn,
        true_w=TRUE_W,
        eval_server=eval_server,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=EXP_NAME,
        exp_params=PARAMS,
        **EXP_ARGS,
    )

    """ Experiment """
    experiment.update_key(rng_key)
    experiment.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    parser.add_argument("--DESIGNER", action="store_true")
    parser.add_argument("--IRD", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE
    IRD = args.IRD
    DESIGNER = args.DESIGNER

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params("examples/params/universal_template.yaml")
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/universal_params.yaml")
    locals().update(PARAMS)

    for ki in copy.deepcopy(RANDOM_KEYS):
        main(random_key=ki)
