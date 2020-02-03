"""Run Active-IRD Interactive Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""

from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.exps.utils import load_params, examples_dir, data_dir
from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.distrib.particles import ParticleServer
from rdb.infer.ird_oc import IRDOptimalControl
from os.path import join, expanduser
from rdb.optim.mpc import build_mpc
from functools import partial
from rdb.infer import *
from jax import random
import numpyro.distributions as dist
import yaml, argparse


class objectview(object):
    """Load dict params into object, because notebook doesn't support
    `local().update(params)`
    """

    def __init__(self, d):
        self.__dict__ = d


def run_interactive(active_fn_name, random_keys=None, load_design=-1, evaluate=False):
    ## Load parameters
    PARAMS = load_params(join(examples_dir(), "params/interactive_template.yaml"))
    PARAMS["EXP_NAME"] = f"{PARAMS['EXP_NAME']}_{active_fn_name}_design_{load_design}"
    PARAMS["INTERACTIVE_NAME"] = f"{PARAMS['INTERACTIVE_NAME']}_{active_fn_name}"

    ## Override Parameters
    if random_keys is not None:
        PARAMS["RANDOM_KEYS"] = random_keys

    p = objectview(PARAMS)
    print(f"Interactive Experiment Random Key {p.RANDOM_KEYS}")

    ## Previous design_data
    if load_design > 0:
        design_data = load_params(join(examples_dir(), f"designs/{p.ENV_NAME}.yaml"))
        assert design_data["ENV_NAME"] == p.ENV_NAME, "Environment name mismatch"
    else:
        design_data = {}

    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(p.ENV_NAME)
        env.reset()
        return env

    def controller_fn(env):
        controller, runner = build_mpc(
            env, env.main_car.cost_runtime, p.HORIZON, env.dt, replan=False
        )
        return controller, runner

    ## Prior sampling & likelihood functions for PGM
    prior = LogUniformPrior(
        rng_key=None,
        normalized_key=p.NORMALIZED_KEY,
        feature_keys=p.FEATURE_KEYS,
        log_max=p.MAX_WEIGHT,
    )
    ird_proposal = IndGaussianProposal(
        rng_key=None,
        normalized_key=p.NORMALIZED_KEY,
        feature_keys=p.FEATURE_KEYS,
        proposal_var=p.IRD_PROPOSAL_VAR,
    )

    ## Evaluation Server
    eval_server = ParticleServer(
        env_fn, controller_fn, num_workers=p.NUM_EVAL_WORKERS, parallel=p.PARALLEL
    )

    ird_model = IRDOptimalControl(
        rng_key=None,
        env_id=p.ENV_NAME,
        env_fn=env_fn,
        controller_fn=controller_fn,
        eval_server=eval_server,
        beta=p.BETA,
        true_w=None,
        prior=prior,
        num_normalizers=p.NUM_NORMALIZERS,
        normalized_key=p.NORMALIZED_KEY,
        proposal=ird_proposal,
        sample_args={"num_warmups": p.NUM_WARMUPS, "num_samples": p.NUM_SAMPLES},
        interactive_mode=True,
        interactive_name=p.INTERACTIVE_NAME,
        save_root=f"{SAVE_ROOT}/{SAVE_NAME}",
        exp_name=f"{EXP_NAME}",
    )

    """ Active acquisition function for experiment """
    active_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None, model=ird_model, beta=p.BETA, debug=False
        ),
        "ratiomean": ActiveRatioTest(
            rng_key=None, model=ird_model, method="mean", debug=False
        ),
        "ratiomin": ActiveRatioTest(
            rng_key=None, model=ird_model, method="min", debug=False
        ),
        "random": ActiveRandom(rng_key=None, model=ird_model),
    }
    keys = list(active_fns.keys())
    for key in keys:
        if key != active_fn_name:
            del active_fns[key]
    assert len(list(active_fns)) == 1

    # Task Sampling seed
    if p.FIXED_TASK_SEED is not None:
        fixed_task_seed = random.PRNGKey(p.FIXED_TASK_SEED)
    else:
        fixed_task_seed = None

    SAVE_ROOT = data_dir()
    experiment = ExperimentActiveIRD(
        ird_model,
        active_fns,
        eval_server=eval_server,
        iterations=p.EXP_ITERATIONS,
        num_eval_tasks=p.NUM_EVAL_TASKS,
        num_eval_map=p.NUM_EVAL_MAP,
        num_active_tasks=p.NUM_ACTIVE_TASKS,
        num_active_sample=p.NUM_ACTIVE_SAMPLES,
        normalized_key=p.NORMALIZED_KEY,
        fixed_task_seed=fixed_task_seed,
        design_data=design_data,
        num_load_design=load_design,
        save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        exp_name=f"{p.EXP_NAME}",
        exp_params=PARAMS,
    )

    for ki in p.RANDOM_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        if evaluate:
            # Post notebook evaluation
            # experiment.run_evaluation(override=False)
            experiment.run_evaluation(override=True)
        else:
            # Interactive notebook Experiment
            experiment.run(plot_candidates=True)


if __name__ == "__main__":
    # for exp_key in ["infogain", "ratiomean", "ratiomin", "random"]:
    for exp_key in ["infogain", "ratiomean", "ratiomin", "random"]:
        # for exp_key in ["random"]:
        # for exp_key in ["infogain"]:
        # for exp_key in ["ratiomin"]:
        # for exp_key in ["ratiomean"]:
        run_interactive(exp_key, evaluate=True)
