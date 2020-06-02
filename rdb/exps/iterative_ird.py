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
import jax.numpy as np
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
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="mean",
        eval_seed=None,
        # Active sampling
        num_active_tasks=4,
        num_active_sample=-1,
        exp_mode="design",
        # Metadata
        save_root="examples/notebook/test",
        design_root="examples/notebook/test",
        exp_name="iterative_proposal",
        exp_params={},
    ):
        # IRD model
        self._model = model
        self._env_fn = env_fn
        self._active_fns = active_fns
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
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._num_active_sample = num_active_sample
        self._exp_mode = exp_mode
        assert self._exp_mode in {"design", "evaluate"}
        # Save path
        assert (
            "joint" in exp_name
            or "independent" in exp_name
            or "divide" in exp_name
            or "batch" in exp_name
        )
        self._joint_mode = "joint" in exp_name or "batch" in exp_name
        self._exp_params = exp_params
        self._exp_name = exp_name
        self._design_root = design_root
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

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Iterative IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
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
        initial_task = random_choice(self._get_rng_task(), self._train_tasks, 1)[0]
        self._tasks = {key: [initial_task] for key in self._active_fns.keys()}
        self._obs = {key: [] for key in self._active_fns.keys()}
        self._obs_ws = {key: [] for key in self._active_fns.keys()}
        self._beliefs = {key: [] for key in self._active_fns.keys()}
        self._eval_hist = {key: [] for key in self._active_fns.keys()}

    def get_task(self, method):
        """ Main function
        """
        assert method in self._active_fns.keys()
        n_tasks = len(self._tasks[method])
        print(f"Method: task no.{n_tasks}")

        return self._tasks[method][-1]

    def add_obs(self, method, obs_ws):
        """ Main function
        """
        assert method in self._active_fns.keys()
        n_tasks = len(self._tasks[method])
        print(f"Method takes obs no.{n_tasks}")

        ## Record observation
        if len(self._obs_ws[method]) < n_tasks:
            self._obs_ws[method].append(obs_ws)
        else:
            self._obs_ws[method][n_tasks - 1] = obs_ws

    def propose(self):
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
            self._get_rng_task(), self._train_tasks, self._num_active_tasks
        )
        candidate_scores = {}
        for method in self._active_fns.keys():
            self._log_time(f"Running proposal for: {method}")
            next_task = self._propose_task(method, candidates, candidate_scores)
            self._tasks[method].append(next_task)

        self._plot_candidate_scores(candidate_scores)
        self._log_time(f"Proposal finished")
        self._save()

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
            self._num_eval_tasks,
            replacement=False,
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
        while not done:
            for method in self._active_fns.keys():
                self._log_time(f"Loading evaluation for: {method}")
                obs_ws = self._obs_ws[method]
                tasks = self._tasks[method]
                n_tasks = len(obs_ws)
                if wi + 1 > n_tasks:
                    done = True
                elif wi + 1 > 2:
                    done = True
                else:
                    obs = self._model.create_particles(
                        obs_ws[: (wi + 1)],
                        save_name=f"iterative_obs",
                        controller=self._model._designer._sample_controller,
                        runner=self._model._designer._sample_runner,
                    )
                    belief = self._model.sample(
                        onp.array(tasks[: wi + 1]),
                        obs=obs,
                        save_name=f"ird_belief_method_{method}_itr_{wi + 1}",
                    )
                    # belief.load()
                    self._obs[method].append(obs)
                    self._beliefs[method].append(belief)
            wi += 1
        self._log_time(f"Loading finished")

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
        for method in self._active_fns.keys():
            self._log_time(f"Running evaluation for: {method}")
            for belief in self._beliefs[method]:
                if self._eval_method == "uniform":
                    belief_sample = belief.subsample(self._num_eval)
                elif self._eval_method == "map":
                    belief_sample = belief.map_estimate(self._num_eval, log_scale=False)

                print(f"Evaluating method {method} belief: Begin")
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
                self._save_eval(eval_info)

            for obs in self._obs[method]:
                print(f"Evaluating method {method} observation: Begin")
                self._eval_server.compute_tasks(
                    "Evaluation", obs, self._eval_tasks, verbose=True
                )
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
            np.savez(f, **data)

        ## Save user input yaml
        yaml_save = f"{self._save_dir}/yaml/rng_{self._rng_name}_designs.yaml"
        os.makedirs(os.path.dirname(yaml_save), exist_ok=True)
        tasks = {key: [v.tolist() for v in val] for (key, val) in self._tasks.items()}
        # print(self._rng_name)
        with open(yaml_save, "w+") as stream:
            yaml.dump(
                dict(tasks=tasks, designs=self._obs_ws),
                stream,
                default_flow_style=False,
            )

    def _save_eval(self, eval_info):
        npy_path = f"{self._save_dir}/{self._exp_name}_eval_seed_{self._rng_name}.npy"
        data = dict(eval_info=eval_info, eval_tasks=self._eval_tasks)
        np.save(npy_path, eval_info)

    def add_evaluate_obs(self):
        self.reset()
        npy_path = f"{self._save_dir}/{self._exp_name}_eval_seed_{self._rng_name}.npy"
        eval_info = np.load(npy_path, allow_pickle=True).item()

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


def run_iterative(evaluate=False, gcp_mode=False):
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
        PARAMS = load_params(f"{examples_dir()}/params/iterative_template.yaml")
        p = objectview(PARAMS)
        # Copy design yaml data
        # locals().update(PARAMS)
        yaml_dir = f"{examples_dir()}/designs/{p.SAVE_NAME}/{p.EXP_NAME}/yaml"
        os.makedirs(yaml_dir, exist_ok=True)
        if os.path.exists(yaml_dir):
            shutil.rmtree(yaml_dir)
        shutil.copytree(f"{data_dir()}/{p.SAVE_NAME}/{p.EXP_NAME}/yaml", yaml_dir)
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
        exp_mode=exp_mode,
        # Saving
        save_root=f"{SAVE_ROOT}/{p.SAVE_NAME}",
        design_root=f"{DESIGN_ROOT}/{p.SAVE_NAME}",
        exp_name=p.EXP_NAME,
        exp_params=PARAMS,
        **p.EXP_ARGS,
    )
    return experiment
