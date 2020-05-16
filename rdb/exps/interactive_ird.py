"""Interactive-IRD Experiment.

Full Experiment:
    1) (Jupyter notebook): sample n training tasks & designs
    2) (Offline): load designs, evaluate and propose next tasks
    3) (Jupyter notebook): continue design
    4) (Offline): load designs, evaluate

"""

from rdb.visualize.plot import *
from rdb.infer.utils import *
from rdb.exps.utils import *
from tqdm.auto import tqdm
import jax.numpy as np
import numpy as onp
import time
import copy
import yaml
import os


class ExperimentInteractiveIRD(object):
    """Interactive Reward Design Experiment.
    """

    def __init__(
        self,
        model,
        env_fn,
        active_fns,
        eval_server,
        # Evaluation parameters
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="uniform",
        eval_seed=None,
        num_propose=4,
        # Active sampling
        num_active_tasks=4,
        num_active_sample=-1,
        exp_mode="propose",
        # Metadata
        save_root="examples/notebook/test",
        design_root="examples/notebook/test",
        exp_name="interactive_one_proposal",
        test_mode=False,
        exp_params={},
    ):
        # IRD model
        self._model = model
        self._env_fn = env_fn
        self._active_fns = active_fns
        self._eval_server = eval_server
        # Random key & function
        self._rng_key, self._rng_name = None, None
        self._eval_seed = random.PRNGKey(eval_seed)
        # Evaluation
        self._num_eval = num_eval
        self._eval_method = eval_method
        assert eval_method in {"map", "uniform"}
        self._num_eval_tasks = num_eval_tasks
        self._eval_env_name = eval_env_name
        self._num_propose = num_propose
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._exp_mode = exp_mode
        assert self._exp_mode in {"propose", "evaluate", "visualize"}
        # Save path
        assert "batch" in exp_name or "divide" in exp_name
        self._batch_mode = "batch" in exp_name
        self._exp_params = exp_params
        self._exp_name = exp_name
        self._design_root = design_root
        self._test_mode = test_mode
        self._save_root = save_root
        self._design_dir = f"{self._design_root}/{self._exp_name}"
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        self._last_time = time.time()

    def update_key(self, rng_key):
        self._rng_name = self._model.rng_name = f"{rng_key[-1]:02d}"
        self._rng_key, rng_model, rng_active = random.split(rng_key, 3)
        self._model.update_key(rng_model)
        save_params(f"{self._save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        # Active functions
        rng_active_keys = random.split(rng_active, len(list(self._active_fns.keys())))
        for fn, rng_key_fn in zip(list(self._active_fns.values()), rng_active_keys):
            fn.update_key(rng_key_fn)

    def _get_rng_eval(self):
        if self._eval_seed is not None:
            self._eval_seed, rng_task = random.split(self._eval_seed)
        else:
            self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def _get_rng_task(self):
        self._rng_key, rng_task = random.split(self._rng_key)
        return rng_task

    def run(self):
        ## Load
        assert self._rng_name is not None and self._rng_key is not None

        if self._exp_mode == "propose":
            ## Propose new tasks based on training input
            yaml_load = (
                f"{self._design_dir}/yaml/rng_{self._rng_name}_training_input.yaml"
            )
            exp_data = self._load_design(yaml_load)
            npy_path = f"{self._save_dir}/training_rng_{self._rng_name}.npy"
            eval_data = {"training_eval": {}}
            self._eval_training_and_propose(exp_data, eval_data)
            self._save_eval(eval_data, npy_path)
            ## Save
            yaml_save = (
                f"{self._save_dir}/yaml/rng_{self._rng_name}_proposal_tasks.yaml"
            )
            self._save_design(exp_data, yaml_save)

        elif self._exp_mode == "evaluate":
            ## Evaluate on proposed tasks based on proposal input
            yaml_load = (
                f"{self._design_dir}/yaml/rng_{self._rng_name}_proposal_input.yaml"
            )
            exp_data = self._load_design(yaml_load)
            npy_path = f"{self._save_dir}/proposal_rng_{self._rng_name}.npy"
            eval_data = {"proposal_eval": {}}
            self._eval_proposal(exp_data, eval_data)
            self._save_eval(eval_data, npy_path)

        elif self._exp_mode == "visualize":
            ## Visualize on proposed tasks based on training input
            yaml_load = (
                f"{self._design_dir}/yaml/rng_{self._rng_name}_proposal_tasks.yaml"
            )
            exp_data = self._load_design(yaml_load)
            npy_path = f"{self._save_dir}/visualize_rng_{self._rng_name}.npy"
            eval_data = {"proposal_eval": {}}
            self._visualize_training_on_proposal(exp_data, eval_data)
            self._save_eval(eval_data, npy_path)

    def _eval_training_and_propose(self, exp_data, eval_data):
        ## Run Active IRD
        self._ird_beliefs = {}
        self._ird_tasks = {}
        self._ird_obs = {}
        self._log_time("IRD Begin")
        for fn_key in self._active_fns.keys():
            training_tasks = exp_data["training_tasks"]
            training_ws = exp_data["training_weights"]
            num_ws = len(exp_data["training_weights"])
            if self._batch_mode:
                training_ws = [exp_data["training_weights"][-1]] * num_ws
            obs = self._model.create_particles(
                training_ws,
                save_name=f"interactive_training_obs",
                controller=self._model._designer._sample_controller,
                runner=self._model._designer._sample_runner,
            )
            belief = self._model.sample(
                tasks=onp.array(training_tasks),
                obs=obs,
                save_name=f"ird_belief_training_method_{fn_key}",
            )
            self._ird_obs[fn_key] = obs
            self._ird_tasks[fn_key] = training_tasks
            self._ird_beliefs[fn_key] = belief
            self._log_time(f"IRD Method: {fn_key}")

        ## Eval
        eval_env = self._env_fn(self._eval_env_name)
        self._eval_tasks = random_choice(
            self._get_rng_eval(),
            eval_env.all_tasks,
            self._num_eval_tasks,
            replacement=False,
        )
        self._train_tasks = self._model.env.all_tasks
        self._log_time(f"Eval Begin")
        for fn_key in self._active_fns.keys():
            eval_info = self._evaluate(
                fn_key,
                self._ird_beliefs[fn_key],
                self._eval_tasks,
                self._eval_method,
                self._num_eval,
            )
            eval_data["training_eval"][fn_key] = eval_info
            self._log_time(f"IRD Method: {fn_key} Eval")

        ## Propose
        candidates = random_choice(
            self._get_rng_task(), self._train_tasks, self._num_active_tasks
        )
        candidate_scores = {}
        self._log_time(f"Propose Begin")
        for fn_key in self._active_fns.keys():
            belief = self._ird_beliefs[fn_key].subsample(self._num_active_sample)
            obs = self._ird_obs[fn_key]
            tasks = self._ird_tasks[fn_key]
            if fn_key == "difficult":
                train_difficulties = self._model.env.all_task_difficulties
                N_top = 1000
                difficult_ids = onp.argsort(train_difficulties)[-N_top:]
                difficult_tasks = self._model.env.all_tasks[difficult_ids]
                next_ids = random_choice(
                    self._get_rng_task(),
                    onp.arange(N_top),
                    self._num_propose,
                    replacement=False,
                )
                next_tasks = difficult_tasks[next_ids]
            else:
                next_tasks, next_scores, _, cand_scores = self._propose_task(
                    candidates, belief, obs, tasks, fn_key, self._num_propose
                )
                candidate_scores[fn_key] = cand_scores
            exp_data[f"proposal_tasks"][fn_key] = next_tasks.tolist()
            self._log_time(f"Propose Method: {fn_key}")
        self._plot_candidate_scores(candidate_scores)

    def _eval_proposal(self, exp_data, eval_data):
        ## Run Active IRD
        self._ird_beliefs = {key: [] for key in self._active_fns.keys()}
        self._ird_tasks = {key: [] for key in self._active_fns.keys()}
        self._ird_obs = {key: [] for key in self._active_fns.keys()}
        self._log_time("IRD Begin")
        for fn_key in self._active_fns.keys():
            training_tasks = exp_data["training_tasks"]
            proposal_tasks = exp_data["proposal_tasks"][fn_key]
            training_ws = exp_data["training_weights"]
            proposal_ws = exp_data["proposal_weights"][fn_key]
            all_tasks = training_tasks + proposal_tasks
            all_ws = training_ws + proposal_ws
            num_ws = len(all_ws)
            if self._batch_mode:
                all_ws = [all_ws[-1]] * num_ws
            obs = self._model.create_particles(
                all_ws,
                save_name=f"interactive_proposal_obs",
                controller=self._model._designer._sample_controller,
                runner=self._model._designer._sample_runner,
            )
            belief = self._model.sample(
                tasks=onp.array(all_tasks),
                obs=obs,
                save_name=f"ird_belief_proposal_method_{fn_key}",
            )
            self._ird_obs[fn_key].append(obs)
            self._ird_tasks[fn_key].append(all_tasks)
            self._ird_beliefs[fn_key].append(belief)
            self._log_time(f"IRD Method: {fn_key}")

        ## Eval
        eval_env = self._env_fn(self._eval_env_name)
        self._eval_tasks = random_choice(
            self._get_rng_eval(), eval_env.all_tasks, self._num_eval, replacement=False
        )
        self._log_time(f"Eval Begin")
        for fn_key in self._active_fns.keys():
            eval_data["proposal_eval"][fn_key] = []
            num_bs = len(self._ird_beliefs[fn_key])
            for bi, belief in enumerate(self._ird_beliefs[fn_key]):
                eval_info = self._evaluate(
                    fn_key, belief, self._eval_tasks, self._eval_method, self._num_eval
                )
                eval_data["proposal_eval"][fn_key].append(eval_info)
                self._log_time(f"IRD Method: {fn_key} Eval ({bi + 1}/{num_bs})")

    def _visualize_training_on_proposal(self, exp_data, eval_data):
        ## Run Active IRD
        self._ird_beliefs = {}
        self._ird_tasks = {}
        self._ird_obs = {}
        self._log_time("IRD Begin")
        for fn_key in self._active_fns.keys():
            training_tasks = exp_data["training_tasks"]
            proposal_tasks = exp_data["proposal_tasks"][fn_key]
            training_ws = exp_data["training_weights"]
            proposal_ws = exp_data["proposal_weights"][fn_key]
            self._ird_beliefs[fn_key] = None
            self._ird_tasks[fn_key] = None
            self._ird_obs[fn_key] = None

            if self._batch_mode:
                training_ws = [training_ws[-1]] * len(training_tasks)
            obs = self._model.create_particles(
                training_ws,
                save_name=f"interactive_proposal_obs",
                controller=self._model._designer._sample_controller,
                runner=self._model._designer._sample_runner,
            )
            belief = self._model.sample(
                tasks=onp.array(training_tasks),
                obs=obs,
                save_name=f"ird_belief_proposal_method_{fn_key}",
            )
            self._ird_obs[fn_key] = obs
            self._ird_tasks[fn_key] = training_ws
            self._ird_beliefs[fn_key] = belief
            self._log_time(f"IRD Method: {fn_key}")

        ## Eval
        self._log_time(f"Visualize Begin")
        num_map_visualize = 8
        runner = self._model._designer._sample_runner
        controller = self._model._designer._sample_controller
        viz_dir = f"{self._save_dir}/visualize"
        os.makedirs(viz_dir, exist_ok=True)

        for fn_key in self._active_fns.keys():
            belief_map = self._ird_beliefs[fn_key].map_estimate(num_map_visualize)
            for wi, ws in enumerate(belief_map.weights):
                for ti, task in enumerate(exp_data[f"proposal_tasks"][fn_key]):
                    self._model.env.set_task(task)
                    self._model.env.reset()
                    state = self._model.env.state
                    actions = controller(state, weights=ws, batch=False)
                    viz_path = f"{viz_dir}/rng_{self._rng_name}_method_{fn_key}_task_{ti:02d}_map_{wi:02d}.mp4"
                    text = f"Method:{fn_key} prior: {len(training_tasks)}"
                    runner.collect_mp4(state, actions, path=viz_path, text=text)
                    print(f"Saved to {viz_path}")
            self._log_time(f"Visualize {fn_key} Done")

    def _load_design(self, load_path):
        """
        Load previous user design yaml data.

        Sample load path:
            dir: data/200410/interactive_proposal_training_03
            filename: interactive_proposal_training_03_rng_02.yaml

        """
        ## Load yaml file for user design
        with open(load_path, "r") as stream:
            exp_data = yaml.safe_load(stream)

        if "proposal_tasks" not in exp_data:
            exp_data["proposal_tasks"] = {key: [] for key in self._active_fns.keys()}
        if "proposal_weights" not in exp_data:
            exp_data["proposal_weights"] = {key: [] for key in self._active_fns.keys()}

        ## Some checking
        # assert set(self._active_fns.keys()) == set(exp_data["proposal_tasks"])
        # assert not self._propose_next or all(
        #     [len(tasks) == 0 for key, tasks in exp_data["proposal_tasks"].items()]
        # )

        print(f"Loaded from {load_path}")
        return exp_data

    def _save_design(self, data, save_path):
        """
        Save user design yaml data.
        """
        if self._test_mode:
            save_path = save_path.replace("yaml/", "yaml_test/")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w+") as stream:
            yaml.dump(data, stream, default_flow_style=False)
        print(f"Saved design to {save_path}")

    def _save_eval(self, data, save_path):
        """
        Save evaluation npy data.
        """
        np.save(save_path, data)
        print(f"Saved evaluation results to {save_path}")

    def _evaluate(self, fn_key, belief, eval_tasks, method, num_samples):
        """
        Evaluate current sampled belief on eval task.

        Note:
            self._num_eval: number of sub samples for evaluation
            self._eval_method: use MAP/uniform sample

        Criteria:
            * Relative Reward.
            * Violations.

        """
        # Compute belief features
        if method == "uniform":
            belief_sample = belief.subsample(num_samples)
        elif method == "map":
            belief_sample = belief.map_estimate(num_samples, log_scale=False)

        print(f"Evaluating method {fn_key}: Begin")
        self._eval_server.compute_tasks(
            "Evaluation", belief_sample, eval_tasks, verbose=True
        )

        desc = f"Evaluating method {fn_key}"
        # (DictList): nvios * (ntasks, nparticles)
        feats_vios = belief_sample.get_violations(eval_tasks)
        feats_vios_arr = feats_vios.onp_array()
        avg_violate = feats_vios_arr.sum(axis=0).mean()
        print(f"    Average Violation {avg_violate:.2f}")
        # import pdb; pdb.set_trace()
        info = {
            "violation": avg_violate,
            "feats_violation": dict(feats_vios.mean(axis=(0, 1))),
        }
        return info

    def _propose_task(
        self, candidates, belief, hist_obs, hist_tasks, fn_key, num_propose
    ):
        """Find best next task for this active function.

        Computation: n_particles(~1k) * n_active(~100) tasks

        Args:
            candidates (list): potential next tasks
            hist_beliefs (list): all beliefs so far, usually only the last one is useful
            hist_obs (Particles.weights[1]): all observations so far
            hist_tasks (list): all tasks proposed so far
            fn_key (str): acquisition function key

        Return:
            next_tasks (np.array): list of tasks
            next_scores (np.array): list of scores
            next_idxs (np.array): list of indices
            scores (np.array): all candidate scores

        """

        assert len(hist_obs) == len(hist_tasks), "Observation and tasks mismatch"
        assert len(hist_obs) > 0, "Need at least 1 observation"
        # Compute belief features
        active_fn = self._active_fns[fn_key]
        print(f"Active proposal method {fn_key}: Begin")
        if fn_key != "random":
            ## =============== Pre-empt heavy computations =====================
            self._eval_server.compute_tasks("Active", belief, candidates, verbose=True)
            self._eval_server.compute_tasks(
                "Active", hist_obs[-1], candidates, verbose=True
            )

        scores = []
        desc = "Evaluaitng candidate tasks"
        feats_keys = self._model._env.features_keys
        for next_task in tqdm(candidates, desc=desc):
            scores.append(
                active_fn(
                    onp.array([next_task]), belief, hist_obs, feats_keys, verbose=False
                )
            )
        scores = onp.array(scores)

        print(f"Function {fn_key} chose task {onp.argmax(scores)} among {len(scores)}")
        next_idxs = onp.argsort(-1 * scores)[:num_propose]
        next_tasks = candidates[next_idxs]
        next_scores = scores[next_idxs]
        return next_tasks, next_scores, next_idxs, scores

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Interactive IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()

    def _plot_candidate_scores(self, candidate_scores):
        ranking_dir = os.path.join(self._save_root, self._exp_name, "candidates")
        os.makedirs(ranking_dir, exist_ok=True)
        for fn_key in candidate_scores.keys():
            # Check current active function
            cand_scores = onp.array(candidate_scores[fn_key])
            other_scores = []
            other_keys = []
            # Check other active function
            other_scores_all = copy.deepcopy(candidate_scores)
            del other_scores_all[fn_key]
            for key in other_scores_all.keys():
                other_scores.append(other_scores_all[key])
                other_keys.append(key)

            # Ranking plot
            file = f"key_{self._rng_name}_fn_{fn_key}_ranking.png"
            path = os.path.join(ranking_dir, file)
            print(f"Candidate plot saved to {path}")
            plot_rankings(
                cand_scores,
                fn_key,
                other_scores,
                other_keys,
                path=path,
                title=f"Interactive: {fn_key}",
                yrange=[-0.4, 4],
                loc="upper left",
                normalize=True,
                delta=0.8,
                annotate_rankings=True,
            )

            for other_s, other_k in zip(other_scores, other_keys):
                file = f"key_{self._rng_name}_compare_{fn_key}_vs_{other_k}.png"
                path = os.path.join(ranking_dir, file)
                plot_ranking_corrs([cand_scores, other_s], [fn_key, other_k], path=path)
