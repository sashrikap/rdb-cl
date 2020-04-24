"""Run Interactive IRD experiment.

Note:
    * See (rdb.exps.interactive_ird.py) for more details.

"""
import numpyro

numpyro.set_host_device_count(3)

from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.exps.utils import load_params, examples_dir, data_dir
from rdb.exps.interactive_ird import ExperimentInteractiveIRD
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from os.path import join, expanduser
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse
import shutil
import ray


def main():
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"  # Don'tchange this line
    DESIGN_ROOT = f"examples/designs" if not GCP_MODE else f"./rdb/examples/designs"
    DEBUG_ROOT = data_dir() if not GCP_MODE else "/gcp_input"

    def env_fn(env_name=None):
        import gym, rdb.envs.drive2d

        if env_name is None:
            env_name = ENV_NAME
        env = gym.make(env_name)
        env.reset()
        return env

    def ird_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            replan=False,
            name=name,
            **IRD_CONTROLLER_ARGS,
        )
        return controller, runner

    def designer_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            replan=False,
            name=name,
            **DESIGNER_CONTROLLER_ARGS,
        )
        return controller, runner

    eval_server = ParticleServer(
        env_fn,
        ird_controller_fn,
        parallel=EVAL_ARGS["parallel"],
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        max_batch=EVAL_ARGS["max_batch"],
    )
    if not ONLY_VISUALIZE:
        eval_server.register("Evaluation", EVAL_ARGS["num_eval_workers"])
    if not ONLY_EVALUATE and not ONLY_VISUALIZE:
        eval_server.register("Active", EVAL_ARGS["num_active_workers"])
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
        exp_name=f"{PARAMS['EXP_NAME']}",
        **IRD_ARGS,
    )

    ## Active acquisition function for experiment
    ACTIVE_BETA = IRD_ARGS["beta"]
    active_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None, beta=ACTIVE_BETA, weight_params=WEIGHT_PARAMS, debug=False
        ),
        "ratiomean": ActiveRatioTest(
            rng_key=None, beta=ACTIVE_BETA, method="mean", debug=False
        ),
        "ratiomin": ActiveRatioTest(
            rng_key=None, beta=ACTIVE_BETA, method="min", debug=False
        ),
        "random": ActiveRandom(rng_key=None),
        "difficult": ActiveRandom(rng_key=None),
    }
    for key in list(active_fns.keys()):
        if key not in ACTIVE_ARGS["active_fns"]:
            del active_fns[key]

    EXP_ARGS["eval_seed"] = random.PRNGKey(EXP_ARGS["eval_seed"])
    if ONLY_EVALUATE:
        exp_mode = "evaluate"
    elif ONLY_VISUALIZE:
        exp_mode = "visualize"
    else:
        exp_mode = "propose"
    experiment = ExperimentInteractiveIRD(
        ird_model,
        env_fn=env_fn,
        active_fns=active_fns,
        eval_server=eval_server,
        exp_mode=exp_mode,
        # Saving
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        design_root=f"{DESIGN_ROOT}/{SAVE_NAME}",
        exp_name=EXP_NAME,
        exp_params=PARAMS,
        **EXP_ARGS,
    )

    """ Experiment """
    for random_key in RANDOM_KEYS:
        rng_key = random.PRNGKey(random_key)
        experiment.update_key(rng_key)
        experiment.run()
    ray.shutdown()  # Prepare for next run, which reinitialize ray with different seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")
    parser.add_argument("--GCP_MODE", action="store_true")
    parser.add_argument("--ONLY_EVALUATE", action="store_true")
    parser.add_argument("--ONLY_VISUALIZE", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE
    ONLY_EVALUATE = args.ONLY_EVALUATE
    ONLY_VISUALIZE = args.ONLY_VISUALIZE

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params("examples/params/interactive_template.yaml")
        # Copy design yaml data
        locals().update(PARAMS)
        yaml_dir = f"examples/designs/{SAVE_NAME}/{EXP_NAME}/yaml"
        os.makedirs(yaml_dir, exist_ok=True)
        if os.path.exists(yaml_dir):
            shutil.rmtree(yaml_dir)
        shutil.copytree(f"data/{SAVE_NAME}/{EXP_NAME}/yaml", yaml_dir)
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/interactive_params.yaml")
        locals().update(PARAMS)
    if not GCP_MODE:
        NUM_EVAL_WORKERS = 4

    main()
