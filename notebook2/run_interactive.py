"""Run Active-IRD Interactive Loop.

Note:
    * See (rdb.exps.active_ird.py) for more details.

"""

from rdb.exps.active_ird import ExperimentActiveIRD
from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
from rdb.infer.ird_oc import IRDOptimalControl
from rdb.optim.mpc import shooting_method
from rdb.distrib.particles import ParticleServer
from rdb.infer.utils import *
from rdb.exps.utils import load_params
from functools import partial
from jax import random
import numpyro.distributions as dist
import yaml, argparse


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def run_interactive(active_fn_name, evaluate=False):
    # Load parameters
    # nb directory: "/Users/jerry/Dropbox/Projects/SafeRew/rdb/examples/notebook"
    if evaluate:
        PARAMS = load_params("examples/params/interactive_template.yaml")
    else:
        # From jupyter notebook
        PARAMS = load_params("../../examples/params/interactive_template.yaml")

    PARAMS["INTERACTIVE_NAME"] = "Jerry_02"
    PARAMS["EXP_NAME"] = f"{PARAMS['EXP_NAME']}_{active_fn_name}"
    PARAMS["INTERACTIVE_NAME"] = f"{PARAMS['INTERACTIVE_NAME']}_{active_fn_name}"
    p = objectview(PARAMS)

    def env_fn():
        import gym, rdb.envs.drive2d

        env = gym.make(p.ENV_NAME)
        env.reset()
        return env

    def controller_fn(env):
        controller, runner = shooting_method(
            env, env.main_car.cost_runtime, p.HORIZON, env.dt, replan=False
        )
        return controller, runner

    """ Prior sampling & likelihood functions for PGM """
    log_prior_dict = {
        "dist_cars": dist.Uniform(-0.05, 0.05),
        "dist_lanes": dist.Uniform(-p.MAX_WEIGHT, p.MAX_WEIGHT),
        "dist_fences": dist.Uniform(-p.MAX_WEIGHT, p.MAX_WEIGHT),
        "dist_objects": dist.Uniform(-p.MAX_WEIGHT, p.MAX_WEIGHT),
        "speed": dist.Uniform(-p.MAX_WEIGHT, p.MAX_WEIGHT),
        "control": dist.Uniform(-p.MAX_WEIGHT, p.MAX_WEIGHT),
    }
    proposal_std_dict = {
        "dist_cars": 1e-6,
        "dist_lanes": p.PROPOSAL_VAR,
        "dist_fences": p.PROPOSAL_VAR,
        "dist_objects": p.PROPOSAL_VAR,
        "speed": p.PROPOSAL_VAR,
        "control": p.PROPOSAL_VAR,
    }
    prior_log_prob_fn = build_log_prob_fn(log_prior_dict)
    prior_sample_fn = build_prior_sample_fn(log_prior_dict)
    proposal_fn = build_gaussian_proposal(proposal_std_dict)
    norm_sample_fn = build_normalizer_sampler(prior_sample_fn, p.NUM_NORMALIZERS)

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
        prior_log_prob_fn=prior_log_prob_fn,
        normalizer_fn=norm_sample_fn,
        proposal_fn=proposal_fn,
        sample_args={"num_warmups": p.NUM_WARMUPS, "num_samples": p.NUM_SAMPLES},
        designer_args={
            "num_warmups": p.NUM_DESIGNER_WARMUPS,
            "num_samples": p.NUM_DESIGNERS,
        },
        interactive_mode=True,
        interactive_name=p.INTERACTIVE_NAME,
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

    if evaluate:
        SAVE_ROOT = "data"
    else:
        # Jupyter notebook
        SAVE_ROOT = "../../data"
    experiment = ExperimentActiveIRD(
        ird_model,
        active_fns,
        eval_server=eval_server,
        iterations=p.EXP_ITERATIONS,
        num_eval_tasks=p.NUM_EVAL_TASKS,
        num_eval_sample=p.NUM_EVAL_SAMPLES,
        num_eval_map=p.NUM_EVAL_MAP,
        num_active_tasks=p.NUM_ACTIVE_TASKS,
        num_active_sample=p.NUM_ACTIVE_SAMPLES,
        max_visualize_weights=p.MAX_WEIGHT,
        save_dir=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        exp_name=f"{p.EXP_NAME}",
        exp_params=PARAMS,
    )

    for ki in p.RANDOM_KEYS:
        key = random.PRNGKey(ki)
        experiment.update_key(key)
        if evaluate:
            # Post-hoc evaluation
            # experiment.run_evaluation(override=False)
            experiment.run_evaluation(override=True)
        else:
            # Interactive Experiment
            experiment.run(plot_candidates=True)


if __name__ == "__main__":
    # for exp_key in ["infogain", "ratiomean", "ratiomin", "random"]:
    # for exp_key in ["infogain", "ratiomean", "ratiomin", "random"]:
    # for exp_key in ["random"]:
    for exp_key in ["infogain"]:
        # for exp_key in ["ratiomin"]:
        # for exp_key in ["ratiomean"]:
        run_interactive(exp_key, evaluate=True)
