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
PORT = 5000
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
    print("Return start_design data:", server_state)
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
        num_initial_tasks=0,
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
        self._num_initial_tasks = num_initial_tasks
        self._initial_tasks = None
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
        self._design_dir = f"{design_root}/{exp_name}"
        self._save_root = save_root
        self._save_dir = f"{save_root}/{exp_name}"
        self._mp4_root = mp4_root
        self._mp4_dir = f"{mp4_root}/{exp_name}"
        self._last_time = time.time()

    def update_key(self, rng_key):
        self._rng_name = self._model.rng_name = f"{rng_key[-1]:02d}"
        self._rng_key, rng_model, rng_active = random.split(rng_key, 3)
        self._model.update_key(rng_model)
        save_params(f"{self._save_root}/params_{self._rng_name}.yaml", self._exp_params)
        ## Active functions
        rng_active_keys = random.split(rng_active, len(list(self._active_fns.keys())))
        for fn, rng_key_fn in zip(list(self._active_fns.values()), rng_active_keys):
            fn.update_key(rng_key_fn)

        ## Training and evaluation tasks
        self._train_tasks = self._model.env.all_tasks
        ## Propose first task
        self._initial_tasks = random_choice(
            self._get_rng_task(), self._train_tasks, self._num_initial_tasks
        )

        ## Build server state store
        self._states = {}

        ## Build cache for proposal
        self._obs, self._obs_ws = {}, {}
        self._beliefs = {}
        self._eval_info = {}

    def run(self):
        api.run(host="0.0.0.0", port=PORT)

    ##======================================================##
    ##==================== Server Methods ==================##
    ##======================================================##

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

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Iterative IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()

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
            img_path = (
                f"{user_id}/rng_{self._rng_name}_thumbnail_{itr}_joint_{index}.png"
            )
            save_path = f"{self._mp4_dir}/{img_path}"
            self._runner.collect_thumbnail(self._env.state, path=save_path, close=False)
            thumbnail_urls.append(save_path)
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
            assert self._initial_tasks is not None
            tasks = onp.array(self._initial_tasks)

        ## Initialize design state
        state = self._build_state(user_id)
        self._states[user_id] = state

        ## Initialize cache
        self._obs[user_id] = {method: [] for method in self._active_keys}
        self._obs_ws[user_id] = {method: [] for method in self._active_keys}
        self._beliefs[user_id] = {method: [] for method in self._active_keys}
        self._eval_info[user_id] = {method: [] for method in self._active_keys}

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
            state["curr_tasks"] = [tasks[0].tolist()]
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
            new_tasks = self.propose(user_id)

        state["proposal_iter"] += 1
        itr = state["proposal_iter"]
        for method, task in new_tasks.items():
            urls = self._get_tasks_thumbnail_urls(user_id, itr, [task])
            state["all_proposal_imgs"][method] += urls
            state["all_proposal_tasks"][method] += [list(task)]
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
        self._save(user_id)
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
                state["final_training_weights"] = [dict(weights)] * num_train
                state["is_training"] = False
                state["curr_tasks"] = []
                state["curr_imgs"] = []
                state["curr_mp4s"] = []
            else:
                state["final_training_weights"].append(dict(weights))
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
            state["final_proposal_weights"][method].append(dict(weights))

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

        self._save(user_id)
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
                    state["all_training_weights"][idx].append(dict(weights))
            else:
                state["all_training_weights"][state["training_iter"]].append(
                    dict(weights)
                )
        else:
            method = state["curr_method"]
            state["all_proposal_weights"][method][state["proposal_iter"]].append(
                dict(weights)
            )
        self._save(user_id)
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
            save_path = f"{self._mp4_dir}/{mp4_path}"
            print(f"Getting video {user_id}, iteration={itr}, trial={trial}, {index}")
            if not self._test_mode:
                actions = self._controller(env_state, weights=weights, batch=False)
                traj, cost, info = self._runner(
                    env_state, actions, weights=weights, batch=False
                )
                self._runner.collect_mp4(env_state, actions, path=save_path)
            print(f"Saved video to {save_path}")
        return mp4_path

    ##======================================================##
    ##=================== Utility Methods ==================##
    ##======================================================##

    def _get_rng_eval(self):
        if self._eval_seed is not None:
            self._eval_seed, rng_task = random.split(self._eval_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def _get_rng_task(self):
        self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def propose(self, user_id):
        self._log_time(f"Proposal Started")
        state = self._states[user_id]

        # Gather current tasks and observations
        tasks = self._get_final_tasks(user_id)
        obs_ws = self._get_final_weights(user_id)
        obs_ws_lens = [len(ws) for ws in obs_ws.values()]
        assert all([n == obs_ws_lens[0] for n in obs_ws_lens])

        ## IRD inference
        for method in self._active_fns.keys():
            num_ws = len(obs_ws[method])
            all_ws = obs_ws[method]
            if self._joint_mode:
                all_ws = [all_ws[-1]] * num_ws
            obs = self._model.create_particles(
                all_ws,
                save_name=f"active_user_obs",
                controller=self._model._designer._sample_controller,
                runner=self._model._designer._sample_runner,
            )
            belief = self._model.sample(
                tasks=tasks[method],
                obs=obs,
                save_name=f"ird_belief_method_{method}_itr_{num_ws}",
            )
            if len(self._beliefs[user_id][method]) < num_ws:
                self._beliefs[user_id][method].append(belief)
                self._obs[user_id][method].append(obs)
            else:
                self._beliefs[user_id][method][num_ws - 1] = belief
                self._obs[user_id][method][num_ws - 1] = obs

        ## Propose next task
        candidates = random_choice(
            self._get_rng_task(), self._train_tasks, self._num_active_tasks
        )
        candidate_scores = {}
        proposed_tasks = {method: None for method in self._active_keys}
        for method in self._active_fns.keys():
            self._log_time(f"Running proposal for: {method}")
            next_task = self._propose_task(
                user_id, method, candidates, candidate_scores
            )
            proposed_tasks[method] = next_task
        self._plot_candidate_scores(candidate_scores)
        self._log_time(f"Proposal finished")
        self._save(user_id)
        return proposed_tasks

    def _get_final_tasks(self, user_id):
        state = self._states[user_id]
        tasks = {method: [] for method in self._active_keys}
        for method in self._active_keys:
            tasks[method] = (
                state["all_training_tasks"] + state["all_proposal_tasks"][method]
            )
        return tasks

    def _get_final_weights(self, user_id):
        state = self._states[user_id]
        obs_ws = {method: [] for method in self._active_keys}
        for method in self._active_keys:
            obs_ws[method] = (
                state["final_training_weights"]
                + state["final_proposal_weights"][method]
            )
        return obs_ws

    def _propose_task(self, user_id, method, candidates, candidate_scores):
        state = self._states[user_id]

        tasks = self._get_final_tasks(user_id)
        obs = self._obs[user_id][method]
        belief = self._beliefs[user_id][method][-1].subsample(self._num_active_sample)

        next_task = None
        if method == "difficult":
            train_difficulties = self._model.env.all_task_difficulties
            N_top = 1000
            difficult_ids = onp.argsort(train_difficulties)[-N_top:]
            difficult_tasks = self._model.env.all_tasks[difficult_ids]
            next_id = random_choice(
                self._get_rng_task(),
                onp.arange(N_top),
                self._num_propose,
                replacement=False,
            )[0]
            next_task = difficult_tasks[next_id]
        elif method == "random":
            next_task = random_choice(
                self._get_rng_task(), candidates, self._num_propose, replacement=False
            )[0]
        else:
            ## Pre-empt heavy computations
            self._eval_server.compute_tasks("Active", belief, candidates, verbose=True)
            self._eval_server.compute_tasks("Active", obs[-1], candidates, verbose=True)
            scores = []
            desc = "Evaluaitng candidate tasks"
            feats_keys = self._model._env.features_keys
            for next_task in tqdm(candidates, desc=desc):
                scores.append(
                    self._active_fns[method](
                        onp.array([next_task]), belief, obs, feats_keys, verbose=False
                    )
                )
            scores = onp.array(scores)

            print(
                f"Function {method} chose task {onp.argmax(scores)} among {len(scores)}"
            )
            next_idxs = onp.argsort(-1 * scores)[: self._num_propose]
            next_task = candidates[next_idxs[0]]

            candidate_scores[method] = scores
        return next_task

    def _plot_candidate_scores(self, candidate_scores):
        ranking_dir = os.path.join(self._save_root, self._exp_name, "candidates")
        os.makedirs(ranking_dir, exist_ok=True)
        for method in candidate_scores.keys():
            cand_scores = onp.array(candidate_scores[method])
            other_scores = []
            other_keys = []
            other_scores_all = copy.deepcopy(candidate_scores)
            del other_scores_all[method]
            for key in other_scores_all.keys():
                other_scores.append(other_scores_all[key])
                other_keys.append(key)
            # Ranking plot
            if len(other_keys) > 0:
                file = f"key_{self._rng_name}_fn_{method}_ranking.png"
                path = os.path.join(ranking_dir, file)
                print(f"Candidate plot saved to {path}")
                plot_rankings(
                    cand_scores,
                    method,
                    other_scores,
                    other_keys,
                    path=path,
                    title=f"Iterative: {method}",
                    yrange=[-0.4, 4],
                    loc="upper left",
                    normalize=True,
                    delta=0.8,
                    annotate_rankings=True,
                )
                for other_s, other_k in zip(other_scores, other_keys):
                    file = f"key_{self._rng_name}_compare_{method}_vs_{other_k}.png"
                    path = os.path.join(ranking_dir, file)
                    plot_ranking_corrs(
                        [cand_scores, other_s], [method, other_k], path=path
                    )

    def evaluate(self, user_id):
        state = self._states[user_id]

        # Gather current tasks and observations
        tasks = self._get_final_tasks(user_id)
        obs_ws = self._get_final_weights(user_id)
        obs_ws_lens = [len(ws) for ws in obs_ws.values()]
        assert all([n == obs_ws_lens[0] for n in obs_ws_lens])

        all_obs = {m: [] for m in self._active_keys}
        all_beliefs = {m: [] for m in self._active_keys}

        eval_env = self._env_fn(self._eval_env_name)
        self._eval_tasks = random_choice(
            self._get_rng_eval(),
            eval_env.all_tasks,
            self._num_eval_tasks,
            replacement=False,
        )
        # Load belief
        # npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        yaml_save = f"{self._design_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        with open(yaml_save, "r") as stream:
            hist_data = yaml.safe_load(stream)

        wi = 0
        done = False
        while not done:
            for method in self._active_fns.keys():
                self._log_time(f"Loading evaluation for: {method}")
                n_tasks = len(obs_ws[method])
                if wi + 1 > n_tasks:
                    done = True
                elif wi + 1 > 2:
                    done = True
                else:
                    obs = self._model.create_particles(
                        obs_ws[method][: (wi + 1)],
                        save_name=f"active_user_obs",
                        controller=self._model._designer._sample_controller,
                        runner=self._model._designer._sample_runner,
                    )
                    belief = self._model.sample(
                        onp.array(tasks[method][: wi + 1]),
                        obs=obs,
                        save_name=f"ird_belief_method_{method}_itr_{wi + 1}",
                    )
                    # belief.load()
                    all_obs[method].append(obs)
                    all_beliefs[method].append(belief)
            wi += 1
        self._log_time(f"Loading finished")

        # Compute belief features
        eval_info = {
            key: {"violation": [], "feats_violation": [], "all_violation": []}
            for key in self._active_fns.keys()
        }
        for method in self._active_fns.keys():
            self._log_time(f"Running evaluation for: {method}")
            for belief in all_beliefs[method]:
                if self._eval_method == "mean":
                    belief_sample = belief.subsample(self._num_eval)
                elif self._eval_method == "map":
                    belief_sample = belief.map_estimate(self._num_eval, log_scale=False)

                desc = f"Evaluating method {method}"
                print(f"{desc}: Begin")
                self._eval_server.compute_tasks(
                    "Evaluation", belief_sample, self._eval_tasks, verbose=True
                )
                # (DictList): nvios * (ntasks, nparticles)
                feats_vios = belief_sample.get_violations(self._eval_tasks)
                feats_vios_arr = feats_vios.onp_array()
                avg_violate = feats_vios_arr.sum(axis=0).mean()
                print(f"    Average Violation {avg_violate:.2f}")
                eval_info[method]["violation"].append(avg_violate)
                eval_info[method]["feats_violation"].append(
                    dict(feats_vios.mean(axis=(0, 1)))
                )
                eval_info[method]["all_violation"].append(dict(feats_vios.mean(axis=1)))

                self._eval_info[user_id] = eval_info
                self._save_eval()
        self._log_time(f"Evaluation finished")
        return eval_info

    def _save(self, user_id):
        ## Save beliefs
        for method in self._active_keys:
            for belief in self._beliefs[user_id][method]:
                belief.save()

        ## Save proposed tasks
        data = dict(
            seed=self._rng_name,
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            states=self._states,
        )
        npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        with open(npz_path, "wb+") as f:
            np.savez(f, **data)

        ## Save user input yaml
        yaml_save = f"{self._save_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        os.makedirs(os.path.dirname(yaml_save), exist_ok=True)
        tasks = self._get_final_tasks(user_id)
        obs_ws = self._get_final_weights(user_id)

        with open(yaml_save, "w+") as stream:
            yaml.dump(
                dict(tasks=tasks, designs=obs_ws), stream, default_flow_style=False
            )

    def _save_eval(self):
        npy_path = f"{self._save_dir}/{self._exp_name}_eval_seed_{self._rng_name}.npy"
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        data = dict(eval_info=self._eval_info, eval_tasks=self._eval_tasks)
        np.save(npy_path, data)


def run_experiment_server(
    evaluate=False, gcp_mode=False, test_mode=False, register_both=False
):
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
        if register_both:
            eval_server.register("Evaluation", p.EVAL_ARGS["num_eval_workers"])
            eval_server.register("Active", p.EVAL_ARGS["num_eval_workers"])
        elif evaluate:
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
