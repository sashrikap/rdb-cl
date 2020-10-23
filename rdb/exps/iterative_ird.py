"""Iterative-IRD Experiment.

Full Experiment (Jupyter notebook):
for itr in range(n_iters):
    1) Enter design on proposed task
    2) Propose next task

Full Evaluation (Offline):
for itr in range(n_iters):
    1) Evaluate belief(itr) on evaluation tasks

"""


from rdb.visualize.plot import *
from rdb.exps.utils import *
from tqdm.auto import tqdm
from rdb.infer import *
from jax import random
import jax.numpy as jnp
import numpy as onp
import time
import copy
import yaml
import os


class ExperimentIterativeIRD(object):
    """Iteractive Reward Design.
    """

    def __init__(
        self,
        model,
        env_fn,
        active_fns,
        # Evaluation parameters
        eval_server,
        controller_fn,
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="mean",
        eval_seed=None,
        iterations=5,
        plan_beta=1.0,
        # Active sampling
        initial_tasks_file=False,
        num_active_tasks=4,
        num_active_sample=-1,
        # Experiment settings
        exp_mode="design",
        divide_and_conquer=True,
        default_initial_weights={},
        propose_wait=-1,
        # Metadata
        load_previous=False,
        save_root="examples/notebook/test",
        design_root="examples/notebook/test",
        exp_name="iterative_proposal",
        exp_params={},
        weight_params={},
    ):
        # IRD model
        self._model = model
        self._env_fn = env_fn
        self._active_fns = active_fns
        self._rng_key, self._rng_name = None, None
        self._iterations = iterations
        self._plan_beta = plan_beta
        # Evaluation
        self._eval_seed = random.PRNGKey(eval_seed)
        self._eval_server = eval_server
        self._num_eval = num_eval
        self._eval_method = eval_method
        assert eval_method in {"map", "mean"}
        self._num_eval_tasks = num_eval_tasks
        self._eval_env_name = eval_env_name
        self._num_propose = 1
        self._divide_and_conquer = divide_and_conquer
        # Active Task proposal
        self._load_previous = load_previous
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._exp_mode = exp_mode
        self._initial_tasks_file = initial_tasks_file
        assert self._exp_mode in {"design", "evaluate"}
        # Save path
        self._exp_params = exp_params
        self._weight_params = weight_params
        self._default_initial_weights = default_initial_weights
        self._design_root = design_root
        self._save_root = save_root
        self._last_time = time.time()
        self.set_exp_name(exp_name)
        ## Initialize experiment
        self._design_process = {"idx_training": 0, "idx_trial": 0, "weights": None}
        self._feedback_process = []
        self._propose_wait = propose_wait
        self._prefix = {"mp4": "", "yaml": "", "img": ""}
        self._design_env = env_fn(self._eval_env_name)
        self._design_controller, self._design_runner = controller_fn(self._design_env)

    def set_exp_name(self, exp_name):
        assert (
            "joint" in exp_name
            or "independent" in exp_name
            or "divide" in exp_name
            or "batch" in exp_name
        )
        self._joint_mode = "joint" in exp_name or "batch" in exp_name
        self._exp_name = exp_name
        self._design_dir = f"{self._design_root}/{self._exp_name}"
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        os.makedirs(self._save_dir, exist_ok=True)
        os.makedirs(self._design_dir, exist_ok=True)

    def update_key(self, rng_key_val):
        rng_key = random.PRNGKey(rng_key_val)
        self._rng_name = self._model.rng_name = f"{rng_key[-1]:02d}"
        self._rng_key, rng_model, rng_active = random.split(rng_key, 3)
        self._model.update_key(rng_model)
        save_params(f"{self._save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        # Active functions
        rng_active_keys = random.split(rng_active, len(list(self._active_fns.keys())))
        for fn, rng_key_fn in zip(list(self._active_fns.values()), rng_active_keys):
            fn.update_key(rng_key_fn)
        self._prefix["mp4"] = f"mp4/rng_{rng_key_val:02d}"
        self._prefix["yaml"] = f"yaml/rng_{rng_key_val:02d}"
        self._prefix["img"] = f"img/rng_{rng_key_val:02d}"

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(
                f">>> Iterative IRD {caption} {self._rng_name} Time: {int(h)}h {int(m)}m {s:.2f}s"
            )
        self._last_time = time.time()

    def _get_rng_eval(self):
        if self._eval_seed is not None:
            self._eval_seed, rng_task = random.split(self._eval_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def _get_rng_task(self):
        self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def reset(self):
        ## Training and evaluation tasks
        self._train_tasks = self._model.env.all_tasks
        ## Propose first task
        if self._initial_tasks_file is None:
            initial_task = random_choice(self._get_rng_task(), self._train_tasks, (1,))[
                0
            ]
        else:
            filepath = f"{examples_dir()}/tasks/{self._initial_tasks_file}.yaml"
            initial_task = load_params(filepath)["TASKS"][0]
        self._tasks = {key: [initial_task] for key in self._active_fns.keys()}
        self._obs = {key: [] for key in self._active_fns.keys()}
        self._obs_ws = {key: [] for key in self._active_fns.keys()}
        self._trial_ws = {
            key: [[] for _ in range(self._iterations)]
            for key in self._active_fns.keys()
        }
        self._beliefs = {key: [] for key in self._active_fns.keys()}
        self._eval_hist = {key: [] for key in self._active_fns.keys()}

        if self._load_previous:
            # Load belief
            # npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
            yaml_save = f"{self._design_dir}/yaml/rng_{self._rng_name}_designs.yaml"
            with open(yaml_save, "r") as stream:
                hist_data = yaml.safe_load(stream)
            self._tasks = hist_data["tasks"]

    def get_task(self, method):
        """ Main function
        """
        assert method in self._active_fns.keys()
        n_tasks = len(self._tasks[method])
        print(f"Method: task no.{n_tasks}")

        return self._tasks[method][-1]

    def add_trial(self, method, obs_ws, idx):
        """ User enters a trial.
        """
        assert method in self._active_fns.keys()
        n_tasks = len(self._tasks[method])
        ## Record observation
        if len(self._trial_ws[method]) <= idx:
            self._trial_ws[method].append([])
        self._trial_ws[method][idx].append(copy.deepcopy(obs_ws))

    def add_obs(self, method, obs_ws, idx, verbose=True):
        """ User confirm a design
        """
        assert method in self._active_fns.keys()
        ## Record observation
        if len(self._obs_ws[method]) <= idx:
            self._obs_ws[method].append(obs_ws)
        else:
            self._obs_ws[method][idx] = obs_ws

    def propose(self):
        self._save()
        ## Increase round
        round_not_finished = []
        idx_training = self._design_process["idx_training"]
        for method, tasks in self._tasks.items():
            if len(tasks) != idx_training + 1:
                round_not_finished.append(method)

        if len(round_not_finished) > 0:
            print("Round not finished, please redo the round")
            return

        if self._load_previous:
            ## Task already loaded
            return

        self._log_time(f"Proposal Started")
        all_n_tasks = [len(tasks) for tasks in self._tasks.values()]
        assert all([n == all_n_tasks[0] for n in all_n_tasks])

        ## IRD inference
        for method in self._active_fns.keys():
            num_ws = len(self._obs_ws[method])
            all_ws = self._obs_ws[method]
            if self._joint_mode:
                all_ws = [all_ws[-1]] * num_ws
            obs = self._model.create_particles(
                all_ws,
                save_name=f"iterative_obs",
                controller=self._model._designer._sample_controller,
                runner=self._model._designer._sample_runner,
            )
            belief = self._model.sample(
                tasks=self._tasks[method],
                obs=obs,
                save_name=f"ird_belief_method_{method}_itr_{num_ws}",
            )
            if len(self._beliefs[method]) < num_ws:
                self._beliefs[method].append(belief)
                self._obs[method].append(obs)
            else:
                self._beliefs[method][num_ws - 1] = belief
                self._obs[method][num_ws - 1] = obs

        ## Propose next task
        candidates = random_choice(
            self._get_rng_task(), self._train_tasks, (self._num_active_tasks,)
        )
        candidate_scores = {}
        for method in self._active_fns.keys():
            self._log_time(f"Running proposal for: {method}")
            next_task = self._propose_task(method, candidates, candidate_scores)
            self._tasks[method].append(next_task)

        self._plot_candidate_scores(candidate_scores)
        self._log_time(f"Proposal finished")
        self._save()
        self._design_process["idx_training"] += 1

    def show_proposed_tasks_for_feedback(self):
        from ipywidgets import Layout, Button, Box
        from ipywidgets import Video

        all_feedback_sliders = {}
        all_feedback_questions = {}
        for method in self._active_fns.keys():
            feedback_questions = [
                {
                    "question": [
                        "On a scale of 1 (strongly disagree) to 7 (strongly agree), how much",
                        "do you think the left video shows good driving behavior ",
                        "for autonomous vehicle?",
                    ],
                    "score": None,
                },
                {
                    "question": [
                        "On a scale of 1 (strongly disagree) to 7 (strongly agree), how much",
                        "do you think the left video shows undesirable driving behavior ",
                        "for autonomous vehicle?",
                    ],
                    "score": None,
                },
            ]
            task = self._tasks[method][-1]
            weights = self._obs_ws[method][-1]
            path = self._generate_feedback_visualization(task, weights, method)
            videos = [Video.from_file(path)]
            video_layout = Layout(
                display="flex", flex_flow="column", align_items="stretch", width="100%"
            )
            video_box = Box(children=videos, layout=video_layout)
            # display(video_box)

            all_sliders = []
            all_question_apps = []
            for q in feedback_questions:
                labels = []
                for l in q["question"]:
                    labels.append(widgets.Label(value=l))
                label_box = widgets.VBox(labels)
                slider = widgets.IntSlider(
                    value=4,
                    min=1,
                    max=7,
                    step=1,
                    description="Score:",
                    disabled=False,
                    continuous_update=False,
                    orientation="horizontal",
                    readout=True,
                    readout_format="d",
                )
                slider_layout = Layout(
                    display="flex",
                    flex_flow="row",
                    align_items="stretch",
                    # border='solid',
                    width="100%",
                )
                all_sliders.append(slider)
                slider_box = Box(children=[slider], layout=slider_layout)
                label_and_slider = widgets.VBox(children=[label_box, slider_box])
                all_question_apps.append(label_and_slider)
                # display(label_box)
                # display(slider_box)
            app = widgets.TwoByTwoLayout(
                top_right=all_question_apps[0],
                bottom_left=video_box,
                bottom_right=all_question_apps[1],
                align_items="center",
                height="700px",
            )
            display(app)

            all_feedback_sliders[method] = all_sliders
            all_feedback_questions[method] = feedback_questions

        submit_button = Button(description="Submit", button_style="danger")
        button_items = [submit_button]
        # button_layout = Layout(
        #     display="flex", flex_flow="column", align_items="stretch", width="100%"
        # )
        button_box = widgets.HBox(children=button_items)
        reminder_layout = Layout(height="80px", min_width="200px")
        reminder = Button(
            layout=reminder_layout,
            description="Feedback not submitted!",
            button_style="warning",
        )
        reminder.add_class("reminder")
        reminder_box = widgets.HBox([reminder])

        ## For interactive
        idx_training = self._design_process["idx_training"]

        def _submit_onclick(b):
            nonlocal reminder_box
            nonlocal all_sliders
            reminder_box.close()
            reminder_layout = Layout(height="80px", min_width="200px")
            reminder = Button(
                layout=reminder_layout,
                description="Feedback submitted!",
                button_style="success",
            )
            reminder.add_class("reminder")
            reminder_box = widgets.HBox([reminder])

            for method in self._active_fns.keys():
                for question, feedback in zip(
                    all_feedback_questions[method], all_feedback_sliders[method]
                ):
                    question["score"] = int(feedback.value)
            display(reminder_box)
            if len(self._feedback_process) >= idx_training:
                self._feedback_process[idx_training - 1] = all_feedback_questions
            else:
                self._feedback_process.append(all_feedback_questions)

        submit_button.on_click(_submit_onclick)
        display(button_box)
        display(reminder_box)

    def _generate_feedback_visualization(self, task, weights, method):
        idx_training = self._design_process["idx_training"]
        idx_trial = self._design_process["idx_trial"]
        self._design_env.set_task(task)
        self._design_env.reset()
        state = self._design_env.state
        actions = self._design_controller(
            state, weights=weights, batch=False, verbose=False
        )
        traj, cost, info = self._design_runner(
            state, actions, weights=weights, batch=False, verbose=False
        )
        mp4_path = (
            f"{self._prefix['mp4']}_feedback_{idx_training - 1}_method_{method}.mp4"
        )
        self._design_runner.collect_mp4(state, actions, path=mp4_path)
        return mp4_path

    def _generate_design_visualization(self, task, weights, method):
        idx_training = self._design_process["idx_training"]
        idx_trial = self._design_process["idx_trial"]
        paths = []
        if self._divide_and_conquer:
            ranges = range(idx_training, idx_training + 1)
        else:
            ranges = range(idx_training + 1)
        for _idx_training in ranges:
            self._design_env.set_task(task)
            self._design_env.reset()
            state = self._design_env.state
            actions = self._design_controller(
                state, weights=weights, batch=False, verbose=False
            )
            traj, cost, info = self._design_runner(
                state, actions, weights=weights, batch=False, verbose=False
            )
            mp4_path = f"{self._prefix['mp4']}_training_{idx_training}_method_{method}_trial_{idx_trial}batch_{_idx_training}.mp4"
            self._design_runner.collect_mp4(state, actions, path=mp4_path)
            paths.append(mp4_path)
        self._design_process["idx_trial"] += 1
        # self._design_process["weights"] = None
        return paths

    def query_user_input_from_notebook(self, methods):
        from ipywidgets import Layout, Button, Box
        from ipywidgets import Video

        if isinstance(methods, list):
            method = methods[0]
            task = self.get_task(method)
        else:
            method = methods
            task = self.get_task(method)
            methods = [methods]

        confirmed = False
        display_text = "Current design not submitted"

        input_layout = Layout(
            display="flex", flex_flow="row", align_items="stretch", width="50%"
        )

        input_items = []
        value_items = []
        key_map = {
            "dist_cars": "- stay away from other cars -",
            "dist_lanes": "--------- change lane ---------",
            "dist_fences": "--------- stay on road ---------",
            "dist_objects": "-- stay away from objects ---",
            "speed": "-------- stable speed ----------",
            "control": "--- smooth & comfortable ----",
        }
        for key in self._weight_params["feature_keys"]:
            style = {"description_width": "initial"}
            float_slider = widgets.FloatLogSlider(
                value=self._default_initial_weights[key],
                base=10,
                min=-5,  # max exponent of base
                max=5,  # min exponent of base
                step=0.2,  # exponent step
                description=key_map[key],
                readout_format=".3f",
                layout={"width": "500px"},
                style=style,
            )
            float_text = widgets.BoundedFloatText(
                value=self._default_initial_weights[key],
                min=0,
                max=1e5,
                step=0.1,
                description="Value:",
                disabled=False,
                readout_format=".3f",
            )
            widgets.jslink((float_slider, "value"), (float_text, "value"))
            value_items.append(float_slider)
            float_layout = Layout(
                display="flex",
                flex_flow="row",
                align_items="stretch",
                # border='solid',
                width="100%",
            )
            float_box = Box(children=[float_slider, float_text], layout=float_layout)
            input_items.append(float_box)

        input_layout = Layout(
            display="flex", flex_flow="column", align_items="stretch", width="100%"
        )
        input_box = Box(children=input_items, layout=input_layout)

        submit_button = Button(description="See Result", button_style="danger")
        confirm_button = Button(description="Confirm", button_style="danger")
        button_items = [submit_button, confirm_button]
        # button_layout = Layout(
        #     display="flex", flex_flow="column", align_items="stretch", width="100%"
        # )
        button_box = widgets.HBox(children=button_items)

        # mp4_path = "/Users/jerry/Dropbox/Projects/SafeRew/rdb/data/200927/universal_joint_init1v1_2000_old_eval_risk/mp4/ird_ibeta_1_eval_0_map_00_costs_-5.959.mp4"
        # ui_items = [input_box, video]
        # ui_box = Box(children=ui_items, layout=ui_layout)
        videos = []
        video_layout = Layout(
            display="flex", flex_flow="column", align_items="stretch", width="60%"
        )

        reminder_layout = Layout(height="80px", min_width="200px")
        reminder = Button(
            layout=reminder_layout, description=display_text, button_style="warning"
        )
        reminder.add_class("reminder")
        reminder_box = widgets.HBox([reminder])

        ## For interactive purpose
        idx_training = self._design_process["idx_training"]
        idx_trial = self._design_process["idx_trial"]

        def _submit_onclick(b):
            nonlocal videos
            nonlocal reminder_box
            for video in videos:
                video.close()
            # reminder_box.close()
            ## TODO
            weights = {}
            for key, item in zip(self._weight_params["feature_keys"], value_items):
                weights[key] = item.value
            ## Record weights during design
            trial_methods = methods
            if idx_training == 0:
                trial_methods = list(self._active_fns.keys())
            for m in trial_methods:
                self.add_trial(m, weights, idx_training)
            mp4_paths = self._generate_design_visualization(task, weights, method)
            videos = [Video.from_file(path) for path in mp4_paths]
            video_layout = Layout(
                display="flex", flex_flow="column", align_items="stretch", width="60%"
            )
            video_box = Box(children=videos, layout=video_layout)
            display(video_box)

        def _confirm_onclick(b):
            nonlocal display_text
            nonlocal reminder_box
            weights = {}
            for key, item in zip(self._weight_params["feature_keys"], value_items):
                weights[key] = item.value
            # self._design_process["weights"] = weights
            print(f"Current training {idx_training} trial {idx_trial}")
            # print(f"Current weights {self._design_process['weights']}")
            # weights = self._design_process["weights"]

            reminder_box.close()
            display_text = "Design submitted."
            for m in methods:
                self.add_obs(m, weights, idx=idx_training, verbose=False)
            reminder_box = Button(
                layout=reminder_layout, description=display_text, button_style="success"
            )
            display(reminder_box)

        submit_button.on_click(_submit_onclick)
        confirm_button.on_click(_confirm_onclick)
        display(reminder_box)
        display(input_box)
        display(button_box)

    def _propose_task(self, method, candidates, candidate_scores):
        belief = self._beliefs[method][-1].subsample(self._num_active_sample)
        obs = self._obs[method]
        tasks = self._tasks[method]

        next_task = None
        if method == "difficult":
            train_difficulties = self._model.env.all_task_difficulties
            N_top = 1000
            difficult_ids = onp.argsort(train_difficulties)[-N_top:]
            difficult_tasks = self._model.env.all_tasks[difficult_ids]
            next_id = random_choice(
                self._get_rng_task(),
                onp.arange(N_top),
                (self._num_propose,),
                replace=False,
            )[0]
            next_task = difficult_tasks[next_id]
        elif method == "random":
            next_task = random_choice(
                self._get_rng_task(), candidates, (self._num_propose,), replace=False
            )[0]
        else:
            ## Pre-empt heavy computations
            self._eval_server.compute_tasks(
                "Active", belief, candidates, verbose=True, waittime=self._propose_wait
            )
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
            # Check current active function
            cand_scores = onp.array(candidate_scores[method])
            other_scores = []
            other_keys = []
            # Check other active function
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

    def evaluate(self):
        self.reset()
        eval_env = self._env_fn(self._eval_env_name)
        self._eval_tasks = random_choice(
            self._get_rng_eval(),
            eval_env.all_tasks,
            (self._num_eval_tasks,),
            replace=False,
        )
        print(f"============== Iterative Evaluation {self._rng_name} ===============")
        # Load belief
        # npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        yaml_save = f"{self._design_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        with open(yaml_save, "r") as stream:
            hist_data = yaml.safe_load(stream)

        self._obs_ws = hist_data["designs"]
        self._tasks = hist_data["tasks"]
        # Compute belief features
        eval_info = {
            key: {
                "violation": [],
                "feats_violation": [],
                "all_violation": [],
                "obs_violation": [],
                "obs_feats_violation": [],
                "obs_all_violation": [],
            }
            for key in self._active_fns.keys()
        }

        ## Set planning beta (higher beta for risk averse planning)
        self._model._beta = self._plan_beta

        for itr in range(self._iterations):
            for method in self._active_fns.keys():
                self._log_time(f"Itr {itr} loading evaluation for: {method}")
                obs_ws = self._obs_ws[method]
                tasks = self._tasks[method]
                n_tasks = len(obs_ws)
                obs = self._model.create_particles(
                    obs_ws[: (itr + 1)],
                    save_name=f"iterative_obs",
                    controller=self._model._designer._sample_controller,
                    runner=self._model._designer._sample_runner,
                )

                self._log_time(
                    f"Itr {itr} evaluating method {method} observation: Begin"
                )
                self._eval_server.compute_tasks(
                    "Evaluation", obs, self._eval_tasks, verbose=True
                )
                # (DictList): nvios * (ntasks, nparticles)
                feats_vios = obs.get_violations(self._eval_tasks)
                feats_vios_arr = feats_vios.onp_array()
                avg_violate = feats_vios_arr.sum(axis=0).mean()
                print(f"    Average obs Violation {method}: {avg_violate:.2f}")
                eval_info[method]["obs_violation"].append(avg_violate)
                eval_info[method]["obs_feats_violation"].append(
                    dict(feats_vios.mean(axis=(0, 1)))
                )
                eval_info[method]["obs_all_violation"].append(
                    dict(feats_vios.mean(axis=1))
                )

                belief = self._model.sample(
                    onp.array(tasks[: itr + 1]),
                    obs=obs,
                    save_name=f"ird_belief_method_{method}_itr_{itr + 1}",
                )
                # belief.load()
                self._obs[method].append(obs)
                self._beliefs[method].append(belief)
                self._log_time(f"Itr {itr} loading {method} finished")

                if self._eval_method == "uniform":
                    belief_sample = belief.subsample(self._num_eval)
                elif self._eval_method == "map":
                    belief_sample = belief.map_estimate(self._num_eval, log_scale=False)

                self._log_time(f"Itr {itr} Evaluating method {method} belief: Begin")
                self._eval_server.compute_tasks(
                    "Evaluation", belief_sample, self._eval_tasks, verbose=True
                )
                # (DictList): nvios * (ntasks, nparticles)
                feats_vios = belief_sample.get_violations(self._eval_tasks)
                feats_vios_arr = feats_vios.onp_array()
                avg_violate = feats_vios_arr.sum(axis=0).mean()
                print(f"    Average Violation  {method}: {avg_violate:.2f}")
                eval_info[method]["violation"].append(avg_violate)
                eval_info[method]["feats_violation"].append(
                    dict(feats_vios.mean(axis=(0, 1)))
                )
                eval_info[method]["all_violation"].append(dict(feats_vios.mean(axis=1)))
                self._save_eval(eval_info)

                self._log_time(f"Itr {itr} {method} Evaluation finished")
        return eval_info

    def _save(self):
        ## Save beliefs
        for method in self._active_fns.keys():
            for belief in self._beliefs[method]:
                belief.save()

        ## Save proposed tasks
        data = dict(
            seed=self._rng_name,
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            obs_ws=self._obs_ws,
            tasks=self._tasks,
            eval_hist=self._eval_hist,
        )
        npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        with open(npz_path, "wb+") as f:
            jnp.savez(f, **data)

        ## Save user input yaml
        yaml_save = f"{self._save_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        os.makedirs(os.path.dirname(yaml_save), exist_ok=True)
        tasks = {
            key: [v.tolist() if type(v) is not list else v for v in val]
            for (key, val) in self._tasks.items()
        }
        with open(yaml_save, "w+") as stream:
            yaml.dump(
                dict(tasks=tasks, designs=self._obs_ws),
                stream,
                default_flow_style=False,
            )

        ## Save user trial yaml
        yaml_trial = f"{self._save_dir}/yaml/rng_{self._rng_name}_trials.yaml"
        os.makedirs(os.path.dirname(yaml_trial), exist_ok=True)
        with open(yaml_trial, "w+") as stream:
            yaml.dump(dict(trials=self._trial_ws), stream, default_flow_style=False)

        ## Save user feedbacks
        yaml_feedbacks = f"{self._save_dir}/yaml/rng_{self._rng_name}_feedbacks.yaml"
        os.makedirs(os.path.dirname(yaml_feedbacks), exist_ok=True)
        with open(yaml_feedbacks, "w+") as stream:
            yaml.dump(
                dict(feedbacks=self._feedback_process), stream, default_flow_style=False
            )

    def _save_eval(self, eval_info):
        npy_path = f"{self._save_dir}/{self._exp_name}_eval_seed_{self._rng_name}.npy"
        data = dict(eval_info=eval_info, eval_tasks=self._eval_tasks)
        jnp.save(npy_path, eval_info)

    def add_evaluate_obs(self):
        self.reset()
        npy_path = f"{self._save_dir}/{self._exp_name}_eval_seed_{self._rng_name}.npy"
        eval_info = jnp.load(npy_path, allow_pickle=True).item()

        eval_env = self._env_fn(self._eval_env_name)
        self._eval_tasks = random_choice(
            self._get_rng_eval(),
            eval_env.all_tasks,
            (self._num_eval_tasks,),
            replace=False,
        )
        # Load belief
        # npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        yaml_save = f"{self._design_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        with open(yaml_save, "r") as stream:
            hist_data = yaml.safe_load(stream)

        self._obs_ws = hist_data["designs"]
        self._tasks = hist_data["tasks"]
        wi = 0
        done = False
        for method in self._active_fns.keys():
            self._log_time(f"Loading evaluation for: {method}")
            obs_ws = self._obs_ws[method]
            tasks = self._tasks[method]
            n_tasks = len(obs_ws)
            assert len(obs_ws) == n_tasks
            for wi in range(len(obs_ws)):
                obs = self._model.create_particles(
                    [obs_ws[wi]],
                    save_name=f"iterative_obs_{wi}",
                    controller=self._model._designer._sample_controller,
                    runner=self._model._designer._sample_runner,
                )
                self._obs[method].append(obs)
        self._log_time(f"Loading finished")

        # Compute belief features
        for key in self._active_fns.keys():
            eval_info[key]["obs_violation"] = []
            eval_info[key]["obs_feats_violation"] = []
            eval_info[key]["obs_all_violation"] = []

        for method in self._active_fns.keys():
            self._log_time(f"Running evaluation for: {method}")
            for io, obs in enumerate(self._obs[method]):

                print(f"Evaluating method {method} itr {io}: Begin")
                self._eval_server.compute_tasks(
                    "Evaluation", obs, self._eval_tasks, verbose=True
                )

                desc = f"Evaluating method {method}"
                # (DictList): nvios * (ntasks, nparticles)
                feats_vios = obs.get_violations(self._eval_tasks)
                feats_vios_arr = feats_vios.onp_array()
                avg_violate = feats_vios_arr.sum(axis=0).mean()
                print(f"    Average Violation {avg_violate:.2f}")
                eval_info[method]["obs_violation"].append(avg_violate)
                eval_info[method]["obs_feats_violation"].append(
                    dict(feats_vios.mean(axis=(0, 1)))
                )
                eval_info[method]["obs_all_violation"].append(
                    dict(feats_vios.mean(axis=1))
                )
                self._save_eval(eval_info)
        self._log_time(f"Evaluation finished")


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def run_iterative(evaluate=False, gcp_mode=False, use_local_params=False):
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

    # Load parameters
    if not gcp_mode:
        if use_local_params:
            params_path = os.path.join(os.path.abspath(""), "iterative_template.yaml")
            print(params_path)
        else:
            params_path = f"{examples_dir()}/params/iterative_template.yaml"
        PARAMS = load_params(params_path)
        p = objectview(PARAMS)
        # Copy design yaml data
        # locals().update(PARAMS)
        exp_yaml_dir = f"{examples_dir()}/designs/{p.SAVE_NAME}/{p.EXP_NAME}/yaml"
        exp_data_dir = f"{data_dir()}/{p.SAVE_NAME}/{p.EXP_NAME}/yaml"
        os.makedirs(exp_yaml_dir, exist_ok=True)
        os.makedirs(exp_data_dir, exist_ok=True)
        if os.path.exists(exp_yaml_dir):
            shutil.rmtree(exp_yaml_dir)
        shutil.copytree(exp_data_dir, exp_yaml_dir)
    else:
        PARAMS = load_params("/dar_payload/rdb/examples/params/iterative_params.yaml")
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
    experiment = ExperimentIterativeIRD(
        ird_model,
        env_fn=env_fn,
        active_fns=active_fns,
        eval_server=eval_server,
        controller_fn=designer_controller_fn,
        exp_mode=exp_mode,
        # Saving
        save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        design_root=f"{DESIGN_ROOT}/{p.SAVE_NAME}",
        exp_name=p.EXP_NAME,
        exp_params=PARAMS,
        weight_params=p.WEIGHT_PARAMS,
        **p.EXP_ARGS,
    )
    return experiment
