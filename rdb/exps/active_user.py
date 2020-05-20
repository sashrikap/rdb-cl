"""Active User experiments


Used as standalone server:
    1) Accepts user reward design
    2) Sends mp4/thumbnail
    3) Proposes next/task


"""

from flask import Flask, json, request
from flask_cors import CORS
from threading import Lock
from rdb.visualize.plot import *
from rdb.exps.utils import *
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
import random as orandom
import jax.numpy as np
import numpy as onp
import time
import copy
import yaml
import os

##======================================================##
##===================== Global Vars ====================##
##======================================================##
api = Flask(__name__)
experiment = None
CORS(api)
lock = Lock()

##======================================================##
##========================= API ========================##
##======================================================##


@api.route("/start", methods=["POST"])
def start_design():
    """User input parameters, return # videos to be shown.

    Request:
        info = {
            user_id (str)
            user_state (dict)
        }

    Response:
        response = {
            user_id (str)
            thumbnail_urls (list)
        }

    """
    req_data = request.get_json()
    print("Received start_design data:", req_data)
    server_state = experiment.start_design(user_id=req_data["user_id"])
    return json.dumps(server_state)


@api.route("/next", methods=["POST"])
def next_design():
    """User input parameters, return # videos to be shown.

    Request:
        info = {
            user_id (str)
            user_state (dict)
        }

    """
    req_data = request.get_json()
    print("Received start_design data:", req_data)

    server_state = experiment.next_design(
        user_id=req_data["user_id"], user_state=req_data["state"]
    )
    return json.dumps(server_state)


@api.route("/new_proposal", methods=["POST"])
def new_proposal():
    """User asks server to make new proposals.

    Request:
        info = {
            user_id (str)
            user_state (dict)
        }

    """
    req_data = request.get_json()
    print("Received start_design data:", req_data)

    server_state = experiment.new_proposal(
        user_id=req_data["user_id"], user_state=req_data["state"]
    )
    return json.dumps(server_state)


@api.route("/try", methods=["POST"])
def try_design():
    """User input parameters, return # videos to be shown.

    Request:
        data = {
            user_id (str)
            weights (json)
            user_states (dict)
        }

    """
    req_data = request.get_json()
    print("Received try_design data:", req_data)
    server_state = experiment.try_design(
        user_id=req_data["user_id"],
        weights=req_data["weights"],
        user_state=req_data["state"],
    )

    return json.dumps(server_state)


@api.route("/make_video", methods=["post"])
def make_video():
    """ User query specific video with id

    Request:
        video = {
            user_id (str)
            index (int): video index
            user_state (dict)
        }

    Response:
        response = {
            url (str): video url relative to base url
        }

    """
    print("Received make_video args:", request.args)
    req_data = request.get_json()
    url = experiment.make_video(
        user_id=req_data["user_id"],
        index=int(req_data["index"]),
        user_state=req_data["state"],
    )
    data = {"mp4_url": url}
    return json.dumps(data)


@api.route("/submit", methods=["POST"])
def submit_design():
    req_data = request.get_json()
    server_state = experiment.next_design(
        user_id=req_data["user_id"], user_state=req_data["state"]
    )
    return json.dumps(server_state)


@api.route("/companies", methods=["GET"])
def get_companies():
    companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]
    return json.dumps(companies)


##======================================================##
##===================== Server Class ===================##
##======================================================##


def all_true(dict_):
    return all(list(dict_.values()))


def all_false(dict_):
    return all([not v for v in dict_.values()])


class ExperimentActiveUser(object):
    """Reward Design Backend Server.
    """

    def __init__(
        self,
        model,
        env_fn,
        controller_fn,
        active_fns,
        # Evaluation parameters
        eval_server,
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="mean",
        eval_seed=None,
        test_mode=False,
        # Active sampling
        num_active_tasks=4,
        num_active_sample=-1,
        exp_mode="design",
        # Metadata
        mp4_root="data/server/test",
        save_root="data/server/test",
        design_root="examples/notebook/test",
        exp_name="iterative_proposal",
        exp_params={},
    ):
        # IRD model
        self._model = model
        self._env_fn = env_fn
        self._controller_fn = controller_fn
        self._env = env_fn()
        self._controller, self._runner = controller_fn(self._env)
        self._active_fns = active_fns
        self._active_keys = list(active_fns.keys())
        self._rng_key, self._rng_name = None, None
        # Evaluation
        self._eval_seed = random.PRNGKey(eval_seed)
        self._eval_server = eval_server
        self._num_eval = num_eval
        self._eval_method = eval_method
        assert eval_method in {"map", "mean"}
        self._num_eval_tasks = num_eval_tasks
        self._eval_env_name = eval_env_name
        self._num_propose = 1
        self._test_mode = test_mode
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._exp_mode = exp_mode
        assert self._exp_mode in {"design", "evaluate"}
        # Save path
        assert "independent" in exp_name or "joint" in exp_name
        self._joint_mode = "joint" in exp_name
        self._exp_params = exp_params
        self._exp_name = exp_name
        self._design_root = design_root
        self._save_root = save_root
        self._mp4_root = mp4_root
        self._last_time = time.time()

    def update_key(self, rng_key):
        self._rng_name = self._model.rng_name = f"{rng_key[-1]:02d}"
        self._rng_key, rng_model, rng_active = random.split(rng_key, 3)
        # self._model.update_key(rng_model)
        save_params(f"{self._save_root}/params_{self._rng_name}.yaml", self._exp_params)
        ## Active functions
        # rng_active_keys = random.split(rng_active, len(list(self._active_fns.keys())))
        # for fn, rng_key_fn in zip(list(self._active_fns.values()), rng_active_keys):
        #     fn.update_key(rng_key_fn)

        ## Build cache
        self._states = {}

    def run(self):
        api.run(host="0.0.0.0", port=5000)

    def _build_state(self, user_id):
        active_keys = self._active_keys
        new_state = dict(
            ## Non-changing
            user_id=user_id,
            joint_mode=self._joint_mode,  # (bool)
            active_keys=active_keys,  # [method (str)]
            ## Temporary
            curr_weights=None,  # weights (dict)
            curr_tasks=[],  # [task (list)]
            curr_imgs=[],  # [url (str)]
            curr_mp4s=[],  # [url (str)]
            curr_active_keys=[],  # [method (str)]
            curr_trial=-1,  # (int)
            curr_method_idx=-1,  # (int)
            curr_method=None,  # (int)
            ## Progress tracking
            is_training=True,  # method (str) -> (bool)
            training_iter=0,  # (int)
            proposal_iter=-1,  # (int)
            ## Book keeping
            all_training_weights=[],  # method (str) -> [weights (dict)]
            all_training_tasks=[],  # [task (list)]
            all_training_imgs=[],  # method (str) -> [url (str)]
            all_proposal_weights={
                m: [] for m in active_keys
            },  # method (str) -> [weights (dict)]
            all_proposal_tasks={
                m: [] for m in active_keys
            },  # method (str) -> [task (list)]
            all_proposal_imgs={
                m: [] for m in active_keys
            },  # method (str) -> [url (str)]
            final_training_weights=[],  # method (str) -> [weights (dict)]
            final_proposal_weights={
                m: [] for m in active_keys
            },  # method (str) -> [weights (dict)]
            need_new_proposal=False,  # (bool)
        )
        return new_state

    def _get_tasks_thumbnail_urls(self, user_id, itr, tasks):
        """Generate thumbnails and return urls (relative)
        Input:
            tasks (ndarray)
        Return:
            thumbnail_urls (list)
        """
        thumbnail_urls = []
        for index, task in enumerate(tasks):
            self._env.set_task(task)
            self._env.reset()
            state = self._env.state
            path = f"{user_id}/rng_{self._rng_name}_thumbnail_{itr}_joint_{index}.png"
            self._runner.collect_thumbnail(self._env.state, path=path, close=False)
            thumbnail_urls.append(path)
        return thumbnail_urls

    def start_design(self, user_id):
        """
        Return:
            thumbnail_urls
        """
        if self._test_mode:
            tasks = onp.array(
                [
                    [-1.75, -2.0, -0.16, -1.3, 0.1, 0.4],
                    [-2.0, -1.0, -0.16, 0.4, 0.1, -0.8],
                    [-1.75, -1.75, -0.16, -1.0, 0.1, -1.7],
                ]
            )
        else:
            tasks = onp.array(
                [
                    [-1.75, -2.0, -0.16, -1.3, 0.1, 0.4],
                    [-2.0, -1.0, -0.16, 0.4, 0.1, -0.8],
                    [-1.75, -1.75, -0.16, -1.0, 0.1, -1.7],
                ]
            )

        ## Initialize design state
        state = self._build_state(user_id)
        self._states[user_id] = state

        ## Propose initial tasks
        state["training_iter"] = 0
        itr = state["training_iter"]
        urls = self._get_tasks_thumbnail_urls(user_id, itr, tasks)
        state["all_training_tasks"] = tasks.tolist()
        state["all_training_imgs"] = urls
        for idx in range(len(state["all_training_tasks"])):
            state["all_training_weights"].append([])

        if self._joint_mode:
            state["curr_tasks"] = tasks.tolist()
            state["curr_imgs"] = urls
        else:
            state["curr_tasks"] = [tasks[0]]
            state["curr_imgs"] = [urls[0]]

        return state

    def new_proposal(self, user_id, user_state):
        """Trigger task proposal.
        """
        state = self._states[user_id]

        curr_active_keys = copy.deepcopy(self._active_keys)
        orandom.shuffle(curr_active_keys)
        state["curr_active_keys"] = curr_active_keys
        state["curr_method_idx"] = 0
        state["curr_method"] = curr_active_keys[0]
        state["curr_trial"] = -1
        state["need_new_proposal"] = False

        ## Propose new tasks
        if self._test_mode:
            new_tasks = {
                m: [-1.75, -2.0, -0.16, -1.3, 0.1, 0.4] for m in curr_active_keys
            }  # method (str) -> task (list)
        else:
            new_tasks = {
                m: [-1.75, -2.0, -0.16, -1.3, 0.1, 0.4] for m in curr_active_keys
            }  # method (str) -> task (list)

        state["proposal_iter"] += 1
        itr = state["proposal_iter"]
        for method, task in new_tasks.items():
            urls = self._get_tasks_thumbnail_urls(user_id, itr, [task])
            state["all_proposal_imgs"][method] += urls
            state["all_proposal_tasks"][method] += [task]
            state["all_proposal_weights"][method].append([])

        if self._joint_mode:
            state["curr_tasks"] = (
                state["all_training_tasks"] + state["all_proposal_tasks"][method]
            )
            state["curr_imgs"] = (
                state["all_training_imgs"] + state["all_proposal_imgs"][method]
            )
        else:
            state["curr_tasks"] = [state["all_proposal_tasks"][method][itr]]
            state["curr_imgs"] = [state["all_proposal_imgs"][method][itr]]
        return state

    def next_design(self, user_id, user_state):
        """Submits designand ask for next design task.

        If current iteration finishes, either start next iteration
        (independent training) or return "need_new_proposal" =  True
        to kick off `new_proposal` call.

        Before:
            user inputs 1 weights design
        After:
            (1.1) user receives next task & thumbnail url
            (1.2) or user receives state["need_new_proposal"] = True and needs to call
                new_proposal.
        """
        state = self._states[user_id]
        weights = state["curr_weights"]
        state["curr_weights"] = None
        state["curr_trial"] = -1

        need_new_proposal = True
        if state["is_training"]:
            # Training mode
            # Decide if need to transition from training phase to proposal phase
            num_train = len(state["all_training_tasks"])
            state["training_iter"] += 1
            if self._joint_mode:
                need_new_proposal = True
                state["final_training_weights"] = [weights] * num_train
                state["is_training"] = False
                state["curr_tasks"] = []
                state["curr_imgs"] = []
                state["curr_mp4s"] = []
            else:
                state["final_training_weights"].append(weights)
                if state["training_iter"] == num_train:
                    need_new_proposal = True
                    state["is_training"] = False
                    state["curr_tasks"] = []
                    state["curr_imgs"] = []
                    state["curr_mp4s"] = []
                else:
                    need_new_proposal = False
                    idx = state["training_iter"]
                    state["curr_tasks"] = [state["all_training_tasks"][idx]]
                    state["curr_imgs"] = [state["all_training_imgs"][idx]]
                    state["curr_mp4s"] = []
        else:
            # Proposal mode
            method = state["curr_active_keys"][state["curr_method_idx"]]
            state["final_proposal_weights"][method].append(weights)

            if state["curr_method_idx"] + 1 == len(self._active_keys):
                need_new_proposal = True
                state["curr_method"] = None
                state["curr_method_idx"] = -1
            else:
                need_new_proposal = False
                state["curr_method_idx"] += 1
                next_method = state["curr_active_keys"][state["curr_method_idx"]]
                idx = state["proposal_iter"]
                state["curr_method"] = next_method

                if self._joint_mode:
                    state["curr_tasks"] = (
                        state["all_training_tasks"]
                        + state["all_proposal_tasks"][next_method]
                    )
                    state["curr_imgs"] = (
                        state["all_training_imgs"]
                        + state["all_proposal_imgs"][next_method]
                    )
                    state["curr_mp4s"] = []
                else:
                    state["curr_tasks"] = [
                        state["all_proposal_tasks"][next_method][idx]
                    ]
                    state["curr_imgs"] = [state["all_proposal_imgs"][next_method][idx]]
                    state["curr_mp4s"] = []

        state["need_new_proposal"] = need_new_proposal

        return state

    def try_design(self, user_id, weights, user_state):
        """Accpepts user design.

        Before:
            user inputs 1 weights design
        After:
            (1) user receive the number of mp4 videos, and need to call get_video to get each.

        """
        state = self._states[user_id]
        state["curr_mp4s"] = [None] * len(state["curr_tasks"])
        state["curr_weights"] = weights
        state["curr_trial"] += 1
        if state["is_training"]:
            if self._joint_mode:
                for idx in range(len(state["all_training_tasks"])):
                    state["all_training_weights"][idx].append(weights)
            else:
                state["all_training_weights"][state["training_iter"]].append(weights)
        else:
            method = state["curr_method"]
            state["all_proposal_weights"][method][state["proposal_iter"]].append(
                weights
            )
        return state

    def make_video(self, user_id, index, user_state):
        """Generates mp4 visualization for user specified rewards

        Before:
            user called try_design
        After:
            (1) user receive mp4 url and displayes it

        """

        state = self._states[user_id]
        task = state["curr_tasks"][index]
        method = state["curr_method"]
        trial = state["curr_trial"]
        weights = state["curr_weights"]

        ## Make video for {itr}-{trial}-{index}
        with lock:
            self._env.set_task(task)
            self._env.reset()
            env_state = self._env.state

            if state["is_training"]:
                itr = state["training_iter"]
                mp4_path = f"{user_id}/rng_{self._rng_name}_method_{method}_training_{itr:02d}_trial_{trial:02d}_joint_{index:02d}.mp4"
            else:
                itr = state["proposal_iter"]
                mp4_path = f"{user_id}/rng_{self._rng_name}_method_{method}_proposal_{itr:02d}_trial_{trial:02d}_joint_{index:02d}.mp4"

            state["curr_mp4s"][index] = mp4_path
            save_path = f"{self._mp4_root}/{mp4_path}"
            print(f"Getting video {user_id}, iteration={itr}, trial={trial}, {index}")
            if not self._test_mode:
                actions = self._controller(env_state, weights=weights, batch=False)
                traj, cost, info = self._runner(
                    env_state, actions, weights=weights, batch=False
                )
                self._runner.collect_mp4(env_state, actions, path=save_path)
            print(f"Saved video to {save_path}")
        return mp4_path


def run_experiment_server(evaluate=False, gcp_mode=False, test_mode=False):
    """Iterative Experiment runner.
    """
    from rdb.exps.active import ActiveInfoGain, ActiveRatioTest, ActiveRandom
    from rdb.exps.utils import load_params, examples_dir, data_dir
    from rdb.distrib.particles import ParticleServer
    from rdb.infer.ird_oc import IRDOptimalControl
    from os.path import join, expanduser
    from rdb.optim.mpc import build_mpc
    from functools import partial
    import shutil

    global experiment

    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d

    # Load parameters
    if not gcp_mode:
        PARAMS = load_params(f"{examples_dir()}/params/user_template.yaml")
        p = objectview(PARAMS)
        # Copy design yaml data
        # yaml_dir = f"{examples_dir()}/designs/{p.SAVE_NAME}/{p.EXP_NAME}/yaml"
        # os.makedirs(yaml_dir, exist_ok=True)
        # if os.path.exists(yaml_dir):
        #     shutil.rmtree(yaml_dir)
        # shutil.copytree(f"{data_dir()}/{p.SAVE_NAME}/{p.EXP_NAME}/yaml", yaml_dir)
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/user_params.yaml")
        p = objectview(PARAMS)
        # locals().update(PARAMS)

    SAVE_ROOT = data_dir() if not gcp_mode else "/gcp_output"  # Don'tchange this line
    DESIGN_ROOT = (
        f"{examples_dir()}/designs" if not gcp_mode else f"./rdb/examples/designs"
    )
    DEBUG_ROOT = data_dir() if not gcp_mode else "/gcp_input"

    def env_fn(env_name=None):
        import gym, rdb.envs.drive2d

        if env_name is None:
            env_name = p.ENV_NAME
        env = gym.make(env_name)
        env.reset()
        return env

    def ird_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            name=name,
            **p.IRD_CONTROLLER_ARGS,
        )
        return controller, runner

    def designer_controller_fn(env, name=""):
        controller, runner = build_mpc(
            env,
            env.main_car.cost_runtime,
            dt=env.dt,
            name=name,
            **p.DESIGNER_CONTROLLER_ARGS,
        )
        return controller, runner

    if test_mode:
        eval_server = None
    else:
        eval_server = ParticleServer(
            env_fn,
            ird_controller_fn,
            parallel=p.EVAL_ARGS["parallel"],
            normalized_key=p.WEIGHT_PARAMS["normalized_key"],
            weight_params=p.WEIGHT_PARAMS,
            max_batch=p.EVAL_ARGS["max_batch"],
        )
        if evaluate:
            eval_server.register("Evaluation", p.EVAL_ARGS["num_eval_workers"])
        else:
            eval_server.register("Active", p.EVAL_ARGS["num_active_workers"])
    ## Prior sampling & likelihood functions for PGM
    def prior_fn(name="", feature_keys=p.WEIGHT_PARAMS["feature_keys"]):
        return LogUniformPrior(
            normalized_key=p.WEIGHT_PARAMS["normalized_key"],
            feature_keys=feature_keys,
            log_max=p.WEIGHT_PARAMS["max_weights"],
            name=name,
        )

    def designer_fn():
        designer = Designer(
            env_fn=env_fn,
            controller_fn=designer_controller_fn,
            prior_fn=prior_fn,
            weight_params=p.WEIGHT_PARAMS,
            normalized_key=p.WEIGHT_PARAMS["normalized_key"],
            save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
            exp_name=p.EXP_NAME,
            **p.DESIGNER_ARGS,
        )
        return designer

    designer = designer_fn()
    ird_model = IRDOptimalControl(
        env_id=p.ENV_NAME,
        env_fn=env_fn,
        controller_fn=ird_controller_fn,
        designer=designer,
        prior_fn=prior_fn,
        normalized_key=p.WEIGHT_PARAMS["normalized_key"],
        weight_params=p.WEIGHT_PARAMS,
        save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        exp_name=f"{p.EXP_NAME}",
        **p.IRD_ARGS,
    )

    ## Active acquisition function for experiment
    ACTIVE_BETA = p.IRD_ARGS["beta"]
    active_fns = {
        "infogain": ActiveInfoGain(
            rng_key=None, beta=ACTIVE_BETA, weight_params=p.WEIGHT_PARAMS, debug=False
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
        if key not in p.ACTIVE_ARGS["active_fns"]:
            del active_fns[key]

    if evaluate:
        exp_mode = "evaluate"
    else:
        exp_mode = "design"
    experiment = ExperimentActiveUser(
        ird_model,
        env_fn=env_fn,
        controller_fn=designer_controller_fn,
        active_fns=active_fns,
        eval_server=eval_server,
        exp_mode=exp_mode,
        # Saving
        mp4_root=f"{p.MP4_ROOT}/{p.SAVE_NAME}",
        save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        design_root=f"{DESIGN_ROOT}/{p.SAVE_NAME}",
        exp_name=p.EXP_NAME,
        exp_params=PARAMS,
        test_mode=test_mode,
        **p.EXP_ARGS,
    )

    # Define random key
    rng_key = random.PRNGKey(p.RANDOM_KEYS[0])
    experiment.update_key(rng_key)

    ## Set up experiment server
    return experiment
    # experiment.run()
