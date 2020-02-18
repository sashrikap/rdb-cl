"""Run Active-IRD Experiment Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""
import numpyro

numpyro.set_host_device_count(3)

from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.exps.utils import *
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse, os
import copy
import ray


def main(random_key, evaluate=False):
    SAVE_ROOT = data_dir() if not GCP_MODE else "/gcp_output"  # Don'tchange this line
    DEBUG_ROOT = data_dir() if not GCP_MODE else "/gcp_input"

    # Define random key
    rng_key = random.PRNGKey(random_key)
    if evaluate:
        # Load pre-saved parameters and update
        print(
            f"\n======== Evaluating exp {EXP_ARGS['save_name']} from {SAVE_ROOT}/{EXP_ARGS['save_name']}========\n"
        )
        params_path = f"{SAVE_ROOT}/{EXP_ARGS['save_name']}/{EXP_ARGS['save_name']}/params_{str(rng_key)}.yaml"
        assert os.path.isfile(params_path)
        eval_params = load_params(params_path)
        locals().update(eval_params)

    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(ENV_NAME)
        env.reset()
        return env

    def controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            replan=False,
            name=name,
            **CONTROLLER_ARGS,
        )
        return controller, runner

    ## Evaluation Server
    eval_server = ParticleServer(
        env_fn,
        controller_fn,
        num_workers=EVAL_ARGS["num_eval_workers"],
        parallel=EVAL_ARGS["parallel"],
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        max_batch=EVAL_ARGS["max_batch"],
    )
    # eval_server = None
    ## Prior sampling & likelihood functions for PGM
    def prior_fn(name=""):
        return LogUniformPrior(
            normalized_key=WEIGHT_PARAMS["normalized_key"],
            feature_keys=WEIGHT_PARAMS["feature_keys"],
            log_max=WEIGHT_PARAMS["max_weights"],
            name=name,
        )

    designer = Designer(
        env_fn=env_fn,
        controller_fn=controller_fn,
        prior_fn=prior_fn,
        weight_params=WEIGHT_PARAMS,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        save_root=f"{SAVE_ROOT}/{EXP_ARGS['save_name']}",
        **DESIGNER_ARGS,
    )

    ird_model = IRDOptimalControl(
        env_id=ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        designer=designer,
        prior_fn=prior_fn,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
        weight_params=WEIGHT_PARAMS,
        save_root=f"{SAVE_ROOT}/{EXP_ARGS['save_name']}",
        **IRD_ARGS,
    )
    ## Active acquisition function for experiment
    active_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None,
            model=ird_model,
            beta=DESIGNER_ARGS["beta"],
            params=WEIGHT_PARAMS,
            debug=False,
        ),
        "ratiomean": ActiveRatioTest(
            rng_key=None, model=ird_model, method="mean", debug=False
        ),
        "ratiomin": ActiveRatioTest(
            rng_key=None, model=ird_model, method="min", debug=False
        ),
        "random": ActiveRandom(rng_key=None, model=ird_model),
    }
    for key in list(active_fns.keys()):
        if key not in ACTIVE_ARGS["active_fns"]:
            del active_fns[key]

    ## Task sampling seed
    if FIXED_TASK_SEED is not None:
        fixed_task_seed = random.PRNGKey(FIXED_TASK_SEED)
    else:
        fixed_task_seed = None

    experiment = ExperimentActiveIRD(
        ird_model,
        designer,
        active_fns,
        true_w=TRUE_W,
        eval_server=eval_server,
        iterations=EXP_ARGS["exp_iterations"],
        num_eval_tasks=EXP_ARGS["num_eval_tasks"],
        num_eval_map=EXP_ARGS["num_eval_map"],
        num_active_tasks=ACTIVE_ARGS["num_active_tasks"],
        num_active_sample=ACTIVE_ARGS["num_active_sample"],
        fixed_task_seed=fixed_task_seed,
        save_root=f"{SAVE_ROOT}/{EXP_ARGS['save_name']}",
        exp_name=f"{EXP_ARGS['EXP_NAME']}",
        exp_params=PARAMS,
        normalized_key=WEIGHT_PARAMS["normalized_key"],
    )

    """ Experiment """
    experiment.update_key(rng_key)
    if evaluate:
        experiment.run_evaluation(override=True)
    else:
        experiment.run()
    ray.shutdown()  # Prepare for next run, which reinitialize ray with different seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--GCP_MODE", action="store_true")
    parser.add_argument("--EVALUATE", action="store_true")
    args = parser.parse_args()

    GCP_MODE = args.GCP_MODE
    EVALUATE = args.EVALUATE

    # Load parameters
    if not GCP_MODE:
        PARAMS = load_params("examples/params/active_template.yaml")
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/active_params.yaml")
    locals().update(PARAMS)
    if not GCP_MODE:
        # RANDOM_KEYS = [24]
        NUM_EVAL_WORKERS = 4
    for ki in copy.deepcopy(RANDOM_KEYS):
        main(random_key=ki, evaluate=EVALUATE)
