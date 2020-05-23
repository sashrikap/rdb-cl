"""Active-IRD Main Experiment.

Full Experiment:
    1) Choose True weight, environment
    2) Simulate Designer on env
    3) Infer b(w) over true weight, sample environment
    4) Evaluate current MAP w
    5) Go back to step 2)

How to tune hyperparameters in Active IRD (By Jerry)
    1) IRD_PROPOSAL_VAR and DESIGNER_VAR are most sensitive for sampling
    2) Tune BETA with `rdb/examples/run_dsigner.py`
    3) Tun NUM_WARMUPS/NUM_SAMPLES with different combinations
    4) Control for randomness with `rdb/examples/run_active.py`

How to control randomness (By Jerry)
    1) In general, run with >=4 different seeds
    2) Evaluation: # MAP, # Tasks
    3) Active: # Candidates, # Histogram bins
    4) MH Sampling: # Designer, # IRD Samples, # Warmup

Note:
    * see rdb/exps/active.py for acquisition functions

Credits:
    * Jerry Z. He 2019-2020

"""

from rdb.distrib.designer import DesignerServer
from rdb.visualize.plot import *
from rdb.infer.utils import *
from rdb.exps.utils import *
from tqdm.auto import tqdm
from jax import random
import jax.numpy as np
import numpy as onp
import time
import copy
import yaml
import os


class ExperimentActiveIRD(object):
    """Active Inverse Reward Design Experiment.

    Args:
        active_fns (dict): map name -> active function
        model (object): IRD model
        iteration (int): algorithm iterations
        num_eval (int): how many particles to sample from posterior belief
        eval_method (str): "map" or "mean"
        num_active_tasks (int): # task candidates for active selection
        num_active_sample (int): running acquisition function on belief
            samples is costly, so subsample belief particles

    """

    def __init__(
        self,
        model,
        env_fn,
        designer_fn,
        active_fns,
        true_w,
        eval_server,
        iterations=10,
        # Evaluation parameters
        num_eval_tasks=4,
        num_eval=-1,
        eval_env_name=None,
        eval_method="map",
        eval_seed=None,
        # Observation model
        obs_method="map",
        use_true_w=False,
        num_prior_tasks=0,
        # Active sampling
        num_active_tasks=4,
        num_active_sample=-1,
        # Debugging
        fixed_candidates=None,
        fixed_belief_tasks=None,
        # Metadata
        design_data=None,
        num_load_design=-1,
        save_root="data/active_ird_exp1",
        exp_name="active_ird_exp1",
        exp_params={},
    ):
        # Inverse Reward Design Model
        self._model = model
        self._env_fn = env_fn
        self._true_w = true_w
        self._active_fns = active_fns
        self._eval_server = eval_server
        self._obs_method = obs_method
        # Random key & function
        self._rng_key = None
        self._rng_name = None
        self._eval_seed = random.PRNGKey(eval_seed)
        self._iterations = iterations
        # Designer simulation
        self._num_prior_tasks = num_prior_tasks
        self._use_true_w = use_true_w
        self._designer_server = DesignerServer(designer_fn)
        if not use_true_w:
            self._designer_server.register(len(active_fns))
        # Evaluation
        self._num_eval = num_eval
        self._eval_method = eval_method
        assert eval_method in {"map", "mean", "post_mean"}
        self._num_eval_tasks = num_eval_tasks
        self._eval_env_name = eval_env_name
        # Active Task proposal
        self._num_active_tasks = num_active_tasks
        self._fixed_candidates = fixed_candidates
        self._num_active_sample = num_active_sample
        self._fixed_belief_tasks = fixed_belief_tasks
        # Save path
        self._exp_params = exp_params
        self._save_root = save_root
        self._exp_name = exp_name
        self._save_dir = f"{self._save_root}/{self._exp_name}"
        self._last_time = time.time()
        # Load design and cache
        self._num_load_design = num_load_design
        self._design_data = design_data

    def _build_cache(self):
        """ Build cache data """
        self._initial_task = None
        self._initial_obs = None
        self._active_eval_hist = {}
        self._active_map_vs_obs_hist = {}
        self._hist_tasks = {}
        self._hist_obs, self._hist_beliefs = {}, {}
        self._hist_candidates = []
        self._hist_cand_scores = {}
        self._eval_tasks = []
        self._train_tasks, self._train_difficulties = [], None
        self._curr_belief = {}
        for key in self._active_fns.keys():
            self._active_eval_hist[key] = []
            self._active_map_vs_obs_hist[key] = []
            self._hist_cand_scores[key] = []
            self._hist_tasks[key] = []
            self._hist_obs[key], self._hist_beliefs[key] = [], []

    def update_key(self, rng_key):
        # Set name
        self._rng_name = str(rng_key)
        self._model.rng_name = str(rng_key)
        self._designer_server.set_rng_name(str(rng_key))
        # Model and designer
        self._rng_key, rng_model, rng_designer, rng_choice, rng_active = random.split(
            rng_key, 5
        )
        self._model.update_key(rng_model)
        self._designer_server.update_key(rng_designer)
        self._designer_server.set_true_w(self._true_w)
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

    def run(self, plot_candidates=False):
        """Main function 1: Run experiment from task.

        Args:
            # task (obj): initial task; if None, start from random.
            plot_candidates (bool): plot rankings of proposed candidates

        """
        print(
            f"\n============= Main Experiment ({self._rng_name}): {self._exp_name} ============="
        )
        self._log_time("Begin")
        self._build_cache()
        self._load_design(self._num_load_design, self._design_data)
        eval_env = self._env_fn(self._eval_env_name)
        num_eval = self._num_eval_tasks
        if self._num_eval_tasks > len(eval_env.all_tasks):
            num_eval = -1
        self._eval_tasks = random_choice(
            self._get_rng_eval(), eval_env.all_tasks, num_eval, replacement=False
        )
        self._train_tasks = self._model.env.all_tasks
        prior_tasks = random_choice(
            self._get_rng_eval(),
            self._train_tasks,
            self._num_prior_tasks,
            replacement=False,
        )
        self._designer_server.set_prior_tasks(prior_tasks)

        ### Main Experiment Loop ###
        for itr in range(0, self._iterations):
            ## Candidates for next tasks
            candidates = random_choice(
                self._get_rng_task(), self._train_tasks, self._num_active_tasks
            )
            self._hist_candidates.append(candidates)

            ### Run Active IRD on Candidates ###
            print(f"\nActive IRD ({self._rng_name}) iteration {itr}")
            active_keys = list(self._active_fns.keys())
            for key in active_keys:
                hist_beliefs = self._hist_beliefs[key]
                hist_obs = self._hist_obs[key]
                hist_tasks = self._hist_tasks[key]
                print(f"Method: {key}")

                ## Actively Task Proposal
                if len(hist_obs) == 0:
                    task, scores = self._propose_random_task()
                    if self._initial_task is None:
                        self._initial_task = task
                    else:
                        task = self._initial_task
                elif key == "random":
                    task, scores = self._propose_random_task()
                elif key == "difficult":
                    if self._train_difficulties is None:
                        self._train_difficulties = self._model.env.all_task_difficulties
                    task, scores = self._propose_random_task()
                    task_ids = onp.argsort(self._train_difficulties)[-1000:]
                    difficult_tasks = self._train_tasks[task_ids]
                    task = random_choice(self._get_rng_task(), difficult_tasks, 1)[0]
                else:
                    belief = hist_beliefs[-1].subsample(self._num_active_sample)
                    task, scores = self._propose_task(
                        candidates, belief, hist_obs, hist_tasks, key
                    )

                self._hist_tasks[key].append(task)
                self._hist_cand_scores[key].append(scores)
                self._plot_candidate_scores(itr)
                self._log_time(f"Itr {itr} {key} Propose")
                if len(hist_beliefs) > 0:
                    self._compare_on_next(key, hist_beliefs[-1], hist_obs[-1], task)

            ### Simulate Designer ###
            if self._use_true_w:
                all_obs = [self._designer_server.designer.truth] * len(active_keys)
            else:
                tasks = onp.array(list(self._hist_tasks.values()))
                all_obs = self._designer_server.simulate(
                    onp.array(tasks), methods=active_keys, itr=itr
                )
            for obs, key in zip(all_obs, active_keys):
                self._hist_obs[key].append(obs)
            self._log_time(f"Itr {itr} Designer Simulated")

            ### IRD Sampling w/ Divide And Conquer ###
            for key in active_keys:
                belief = self._model.sample(
                    tasks=onp.array(self._hist_tasks[key]),
                    obs=self._hist_obs[key],
                    save_name=f"ird_belief_method_{key}_itr_{itr:02d}",
                )
                self._hist_beliefs[key].append(belief)
                self._curr_belief[key] = belief
                self._log_time(f"Itr {itr} {key} IRD Divide & Conquer")

                ## Evaluate, plot and Save
                self._evaluate(
                    key, belief, self._eval_tasks, num_samples=self._num_eval
                )
                self._save(itr=itr, fn_key=key)
                self._log_time(f"Itr {itr} {key} Eval & Save")

    def _propose_random_task(self):
        """Used to choose first task, when user initial design is not provided.
        """

        assert (
            self._num_load_design < 0
        ), "Should not propose random task if user design is provided"
        task = random_choice(self._get_rng_task(), self._train_tasks, 1)[0]
        scores = onp.zeros(self._num_active_tasks)
        return task, scores

    def _propose_task(self, candidates, belief, hist_obs, hist_tasks, fn_key):
        """Find best next task for this active function.

        Computation: n_particles(~1k) * n_active(~100) tasks

        Args:
            candidates (list): potential next tasks
            hist_beliefs (list): all beliefs so far, usually only the last one is useful
            hist_obs (Particles.weights[1]): all observations so far
            hist_tasks (list): all tasks proposed so far
            fn_key (str): acquisition function key

        Note:
            * Require `tasks = env.sample_task()`
            * Use small task space to avoid being too slow.

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
        next_task = candidates[onp.argmax(scores)]
        return next_task, scores

    def _evaluate(self, fn_key, belief, eval_tasks, num_samples, cache=True):
        """Evaluate current sampled belief on eval task.

        Computation: n_map(~4) * n_eval(5~10k) tasks

        Note:
            self._num_eval: number of sub samples for evaluation
            self._eval_method: use mean/mean sample

        Criteria:
            * Relative Reward.
            * Violations.

        """

        # Compute belief features
        if self._eval_method == "mean":
            belief_sample = belief.subsample(num_samples)
        elif self._eval_method == "map":
            belief_sample = belief.map_estimate(num_samples, log_scale=False)
        elif self._eval_method == "post_mean":
            weights_mean = belief.weights.mean(axis=0)
            belief_sample = self._model.create_particles(
                [weights_mean],
                save_name=f"ird_prior_sample",
                controller=self._model._sample_controller,
                runner=self._model._sample_runner,
            )

        target = self._designer_server.designer.truth
        print(f"Evaluating method {fn_key}: Begin")
        self._eval_server.compute_tasks(
            "Evaluation", belief_sample, eval_tasks, verbose=True
        )
        self._eval_server.compute_tasks("Evaluation", target, eval_tasks, verbose=True)

        all_violates, rel_violates = [], []
        all_performs, rel_performs = [], []
        feats_violates = []
        desc = f"Evaluating method {fn_key}"
        for task in tqdm(eval_tasks, desc=desc):
            comparisons = belief_sample.compare_with(task, target=target)
            all_performs.append(comparisons["rews"].mean())
            all_violates.append(comparisons["vios"].mean())  # (nweights,) -> (1,)
            rel_performs.append(comparisons["rews_relative"].mean())
            rel_violates.append(comparisons["vios_relative"].mean())
            feats_violates.append(comparisons["vios_by_name"])

        avg_violate = np.mean(np.array(all_violates, dtype=float))
        avg_perform = np.mean(np.array(all_performs, dtype=float))
        avg_rel_violate = np.mean(np.array(rel_violates, dtype=float))
        avg_rel_perform = np.mean(np.array(rel_performs, dtype=float))
        avg_feats_violate = feats_violates[0] * (1 / float(len(eval_tasks)))
        for fv in feats_violates[1:]:
            avg_feats_violate += fv * (1 / float(len(eval_tasks)))
        if target is not None:
            log_prob_true = float(belief.log_prob(target.weights[0]))
        else:
            log_prob_true = 0.0
        print(f"    Average Violation diff {avg_violate:.2f}")
        print(f"    Average Violation rel {avg_rel_violate:.2f}")
        print(f"    Average Performance diff {avg_perform:.2f}")
        print(f"    Average Performance rel {avg_rel_perform:.2f}")
        print(f"    True Weights Log Prob {log_prob_true:.2f}")

        info = {
            "violation": avg_violate,
            "feats_violation": dict(avg_feats_violate),
            "perform": avg_perform,
            "log_prob_true": log_prob_true,
            "rel_violation": avg_rel_violate,
            "rel_perform": avg_rel_perform,
        }
        if cache:
            self._active_eval_hist[fn_key].append(info)
        return info

    def _compare_on_next(
        self, fn_key, belief, joint_obs, next_task, cache=True, sample_method="mean"
    ):
        """Compare observation and MAP estimate on next proposed task"""
        if sample_method == "map":
            belief_sample = belief.map_estimate(self._num_eval, log_scale=False)
        elif sample_method == "mean":
            belief_sample = belief.subsample(self._num_eval)

        print(f"Evaluating method {fn_key} on next task")
        belief_sample.compute_tasks([next_task])
        joint_obs.compute_tasks([next_task])

        target = self._designer_server.designer.truth
        desc = f"Evaluating method {fn_key}"
        belief_comparisons = belief_sample.compare_with(
            next_task, target=target, relative=True
        )
        belief_perform = belief_comparisons["rews"]
        belief_violate = belief_comparisons["vios"]  # (nweights,)
        obs_comparisons = joint_obs.compare_with(
            next_task, target=target, relative=True
        )
        obs_perform = obs_comparisons["rews"]
        obs_violate = obs_comparisons["vios"]

        print(f"    Belief Violation diff {belief_violate.mean():.2f}")
        print(f"    Belief Performance diff {belief_perform.mean():.2f}")
        print(f"    Obs Violation diff {obs_violate.mean():.2f}")
        print(f"    Obs Performance diff {obs_perform.mean():.2f}")
        info = {
            "belief_perform": belief_perform.tolist(),
            "belief_perform_relative": belief_comparisons["rews_relative"],
            "belief_violation": belief_violate.tolist(),
            "belief_violation_relative": belief_comparisons["vios_relative"],
            "obs_perform": obs_perform.tolist(),
            "obs_perform_relative": obs_comparisons["rews_relative"],
            "obs_violation": obs_violate.tolist(),
            "obs_violation_relative": obs_comparisons["vios_relative"],
        }
        if cache:
            self._active_map_vs_obs_hist[fn_key].append(info)

    def _load_design(self, num_load, design_data, cache=True, compute_belief=True):
        """Load previous weight designs before experiment.

        In real applications, usually designers begin with a few (e.g. 5) environments
        they have in mind, and a rough design based on them. We load these designs and
        perform active ird from there.

        Note:
            design_data (dict):
            {
                'ENV_NAME': gym environment name,
                'DESIGNS':
                [ # list of environment & design pairs
                    {
                        TASK: ...,
                        WEIGHTS: {...}
                    },
                ]
            }
            num_load (int): how many previous designs do we use

        """
        if num_load <= 0:
            return
        assert (
            design_data["ENV_NAME"] == self._exp_params["ENV_NAME"]
        ), "Environment name mismatch"
        assert len(design_data["DESIGNS"]) >= num_load, "Not enough designs to load"

        load_designs = design_data["DESIGNS"][:num_load]
        load_obs, load_tasks = [], []
        for i, design_i in enumerate(load_designs):
            # Load previous design
            task = design_i["TASK"]
            weights = design_i["WEIGHTS"]
            obs = self._model.create_particles(
                [weights],
                save_name=f"ird_prior_design_{i:02d}",
                controller=self._designer_server.designer._sample_controller,
                runner=self._designer_server.designer._sample_runner,
            )
            load_obs.append(obs)
            load_tasks.append(task)

            if cache:
                # Cache previous designs for active functions
                for key in self._active_fns.keys():
                    # Make sure nothing cached before design data
                    assert len(self._hist_obs[key]) == i
                    assert len(self._hist_tasks[key]) == i
                    self._hist_obs[key].append(obs)
                    self._hist_tasks[key].append(task)

        print(f"Loaded {len(load_designs)} prior designs.")
        ## Compute IRD Belief based on loaded data
        if compute_belief:
            belief = self._model.sample(
                self._hist_tasks[key],
                obs=self._hist_obs[key],
                save_name=f"ird_prior_belief_{num_load:02d}",
            )
            for i in range(num_load):
                # Pack the same beliefs into belief history
                assert len(self._hist_beliefs[key]) == i
                self._hist_beliefs[key].append(belief)
        return load_obs, load_tasks

    def _save(self, itr, fn_key=None, skip_weights=False):
        """Save checkpoint.

        Format:
            * data/save_root/exp_name/{exp_name}_seed.npz
              - seed (str): rng_key
              - curr_obs (dict): {method: [obs_w] * num_itr}
              - curr_tasks (dict): {method: [task] * num_eval}
              - eval_tasks (dict): {method: [task] * num_eval}
              - eval_hist (dict): {method: [
                    {"violation": ..., "perform": ...}
                )] * num_itr}
            * data/save_root/exp_name/save/weights_seed_method_itr.npz
              - see `rdb.infer.particles.save()`

        """
        print("Saving to:", self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        ## Save experiment parameters
        save_params(f"{self._save_dir}/params_{self._rng_name}.yaml", self._exp_params)
        ## Save experiment history to npz
        np_obs = {}
        for key in self._hist_obs.keys():
            np_obs[key] = [dict(ob.weights[0]) for ob in self._hist_obs[key]]
        data = dict(
            seed=self._rng_name,
            exp_params=self._exp_params,
            env_id=str(self._model.env_id),
            true_w=self._designer_server.designer.true_w,
            curr_obs=np_obs,
            curr_tasks=self._hist_tasks,
            eval_hist=self._active_eval_hist,
            active_map_vs_obs_hist=self._active_map_vs_obs_hist,
            candidate_tasks=self._hist_candidates,
            candidate_scores=self._hist_cand_scores,
            eval_tasks=self._eval_tasks
            if self._num_eval_tasks > 0
            else [],  # do not save when eval onall tasks (too large)
        )
        npz_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npz"
        with open(npz_path, "wb+") as f:
            np.savez(f, **data)
        ## Save evaluation history to yaml
        npy_path = f"{self._save_dir}/{self._exp_name}_seed_{self._rng_name}.npy"
        np.save(npy_path, self._active_eval_hist)
        ## Save active comparison to yaml
        npy_path = f"{self._save_dir}/{self._exp_name}_map_seed_{self._rng_name}.npy"
        np.save(npy_path, self._active_map_vs_obs_hist)

        if fn_key is not None:
            ## Save belief sample information
            true_w = self._designer_server.designer.true_w
            # Ony save last belief, to save time
            itr = len(self._hist_beliefs[fn_key]) - 1
            belief = self._hist_beliefs[fn_key][-1]
            if itr < len(np_obs[fn_key]):
                obs_w = np_obs[fn_key][itr]
            else:
                obs_w = None
            if not skip_weights:
                belief.save()
        print("Saving done")

    def _log_time(self, caption=""):
        if self._last_time is not None:
            secs = time.time() - self._last_time
            h = secs // (60 * 60)
            m = (secs - h * 60 * 60) // 60
            s = secs - (h * 60 * 60) - (m * 60)
            print(f">>> Active IRD {caption} Time: {int(h)}h {int(m)}m {s:.2f}s")
        self._last_time = time.time()

    def _plot_candidate_scores(self, itr, debug_dir=None):
        if debug_dir is None:
            debug_dir = self._save_root
        hist_scores = self._hist_cand_scores
        ranking_dir = os.path.join(debug_dir, self._exp_name, "candidates")
        os.makedirs(ranking_dir, exist_ok=True)
        for fn_key in hist_scores.keys():
            # Check current active function
            if len(hist_scores[fn_key]) <= itr or fn_key == "random":
                continue
            cand_scores = onp.array(hist_scores[fn_key][itr])
            other_scores = []
            other_keys = []
            # Check other active function
            other_scores_all = copy.deepcopy(hist_scores)
            del other_scores_all[fn_key]
            for key in other_scores_all.keys():
                if len(other_scores_all[key]) > itr:
                    other_scores.append(other_scores_all[key][itr])
                    other_keys.append(key)

            # Ranking plot
            file = f"key_{self._rng_name}_cand_itr_{itr}_fn_{fn_key}_ranking.png"
            path = os.path.join(ranking_dir, file)
            print(f"Candidate plot saved to {path}")
            plot_rankings(
                cand_scores,
                fn_key,
                other_scores,
                other_keys,
                path=path,
                title=f"{fn_key}_itr_{itr}",
                yrange=[-0.4, 4],
                loc="upper left",
                normalize=True,
                delta=0.8,
                annotate_rankings=True,
            )

            for other_s, other_k in zip(other_scores, other_keys):
                file = f"key_{self._rng_name}_cand_itr_{itr}_comare_{fn_key}_vs_{other_k}.png"
                path = os.path.join(ranking_dir, file)
                plot_ranking_corrs([cand_scores, other_s], [fn_key, other_k], path=path)
