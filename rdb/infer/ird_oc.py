"""Inverse Reward Design Module for Optimal Control.

Includes:
    [1] Pyro Implementation
    [2] Custon Implementation
Note:
    * We use `init_state/task` interchangably. In drive2d
     `env.set_task(init_state)` specifies a task to run.

"""

import numpyro
import jax
from numpyro.handlers import scale, condition, seed
from rdb.infer.algos import *
from rdb.optim.utils import concate_dict_by_keys
from rdb.infer.utils import logsumexp, collect_features
from tqdm.auto import tqdm, trange
from time import time


class PGM(object):
    """Generic Probabilisitc Graphical Model Class.

    Methods:
        likelihood (fn): p(obs | theta) p(theta)

    """

    def __init__(self, rng_key, kernel, proposal_fn, sample_method, sample_args):
        self._rng_key = rng_key
        self._sampler = self._build_sampler(
            kernel, proposal_fn, sample_method, sample_args
        )

    def update_key(self, rng_key):
        self._rng_key = rng_key
        self._sampler.update_key(rng_key)

    def _build_sampler(self, kernel, proposal_fn, sample_method, sample_args):
        if sample_method == "mh":
            return MetropolisHasting(
                self._rng_key, kernel, proposal_fn=proposal_fn, **sample_args
            )
        else:
            raise NotImplementedError


class IRDOptimalControl(PGM):
    """Inverse Reward Design for Optimal Control.

    Given environment `env`, `planner`, user input w
    infer p(w* | w).

    Notes:
        * Bayesian Terminology: theta -> true w, obs -> proxy w
        * Provides caching functions to avoid costly feature calcultions

    Args:
        controller (fn): controller function,
            `actions = controller(state, w)`
        runner (fn): runner function,
            `traj, cost, info = runner(state, actions)`
        beta (float): temperature param
            `p ~ exp(beta * reward)`
        prior_log_prob (fn): log probability of prior
        normalizer_fn (fn): sample fixed number of normalizers

    Methods:
        sample (fn): sample b(w) given obs_w
        get_samples (fn): get sample features on new task
        update (fn): incorporate user's obs_w
            next likelihood `p(w | obs_w1) p(w | obs_w2)...`
        infer (fn): infer p`(w* | obs)`

    Example:
        >>> ird_oc = IRDOptimalControl(driving_env, runner, beta, prior_fn)
        >>> _, samples = ird_oc.posterior()

    """

    def __init__(
        self,
        rng_key,
        env,
        controller,
        runner,
        beta,
        prior_log_prob,
        normalizer_fn,
        proposal_fn,
        sample_method="mh",
        sample_args={},  # "num_warmups": 100, "num_samples": 200
        designer_args={},
    ):
        self._rng_key = rng_key
        self._controller = controller
        self._runner = runner
        self._env = env
        self._beta = beta
        self._prior_log_prob = prior_log_prob
        self._normalizer_fn = normalizer_fn
        ## Caching normalizers, samples and features
        self._norm_sample_weights = {}
        self._norm_sample_feats = {}
        self._designer_weights = {}
        self._sample_weights = {}
        self._sample_feats = {}
        self._user_actions = {}
        self._user_feats = {}
        ## Likelihood models
        designer_kernel = self._build_designer_kernel(beta)
        self._designer = self._build_sampler(
            designer_kernel, proposal_fn, sample_method, designer_args
        )
        kernel = self._build_kernel(beta)
        super().__init__(rng_key, kernel, proposal_fn, sample_method, sample_args)

    def update_key(self, rng_key):
        """ Update random key """
        super().update_key(rng_key)
        self._designer.update_key(rng_key)
        self._prior_log_prob = seed(self._prior_log_prob, rng_key)
        self._normalizer_fn = seed(self._normalizer_fn, rng_key)

    def get_samples(self, task, task_name):
        """Sample features for belief weights on a task
        """
        assert task_name in self._sample_weights.keys(), f"{task_name} not found"
        weights = self._sample_weights[task_name]

        # Cache features
        if task_name not in self._sample_feats.keys():
            self._cache_samples(task, task_name)
        feats = self._sample_feats[task_name]
        return weights, feats

    def get_features(self, task, task_name, sample_weights):
        """Get features on new task, given weights
        """
        # assert task_name not in self._sample_weights.keys()
        self._sample_weights[task_name] = sample_weights
        if task_name not in self._sample_feats.keys():
            self._cache_samples(task, task_name, verbose=False)
        feats = self._sample_feats[task_name]
        del self._sample_feats[task_name]
        del self._sample_weights[task_name]
        return feats

    def update(self, obs):
        """ Incorporate new observation """
        pass

    def sample_designer(self, task, task_name, true_w, verbose=True):
        """Sample b(obs_w) given true_w"""
        if task_name not in self._sample_weights.keys():
            sample_ws = self._designer.sample(true_w, task=task)
            self._designer_weights[task_name] = sample_ws
        else:
            sample_ws = self._designer_weights[task_name]
        return sample_ws

    def sample(self, task, task_name, obs_w, verbose=True):
        """Sample b(w) for true weights given obs_w."""
        # Cache normalizer samples
        self._cache_normalizer(task, task_name)
        self._cache_user_feats(task, task_name, obs_w)

        # Cache user actions
        if task_name not in self._sample_weights.keys():
            norm_feats = self._norm_sample_feats[task_name]
            user_feats = self._user_feats[task_name]
            sample_ws = self._sampler.sample(
                obs_w,
                norm_sample_feats=norm_feats,
                user_feats=user_feats,
                verbose=verbose,
            )
            self._sample_weights[task_name] = sample_ws
        else:
            sample_ws = self._sample_weights[task_name]
        return sample_ws

    def evaluate(self, task, task_name, sample_ws):
        state = self._get_init_state(task)
        num_violate = 0.0
        for w in sample_ws:
            actions = self._controller(state, weights=w)
            traj, cost, info = self._runner(state, actions, weights=w)
            violations = info["violations"]
            num = sum([sum(v) for v in violations.values()])
            # print(f"violate {num} acs {np.mean(actions):.3f} xs {np.mean(traj):.3f}")
            num_violate += num
        return float(num_violate) / len(sample_ws)

    def _get_init_state(self, task):
        self._env.set_task(task)
        self._env.reset()
        state = self._env.state
        return state

    def _cache_user_feats(self, task, task_name, user_w):
        """For new observation, cache optimal features."""
        if task_name in self._user_feats.keys():
            return

        state = self._get_init_state(task)
        actions = self._controller(state, weights=user_w)
        _, _, info = self._runner(state, actions, weights=user_w)
        feats = info["feats"]
        self._user_actions[task_name] = actions
        self._user_feats[task_name] = feats

    def _cache_samples(self, task, task_name, verbose=True):
        """For new tasks, see how previous samples do.
        """
        assert (
            task_name in self._sample_weights.keys()
        ), f"{task_name} weight samples missing"
        sample_ws = self._sample_weights[task_name]
        state = self._get_init_state(task)
        desc = "Collecting Features for Belief Samples"
        if not verbose:
            desc = None
        sample_feats = collect_features(
            sample_ws, state, self._controller, self._runner, desc=desc
        )
        self._sample_feats[task_name] = sample_feats

    def _cache_normalizer(self, task, task_name):
        """For new tasks, need to build normalizer.

        Normalizer cached in `self._norm_sample_weights` and
        `self._norm_sample_feats`

        """
        if task_name in self._norm_sample_weights.keys():
            return

        norm_ws = self._normalizer_fn()
        state = self._get_init_state(task)
        norm_feats = collect_features(
            norm_ws,
            state,
            self._controller,
            self._runner,
            desc="Collecting Normalizer Samples",
        )
        norm_feats = concate_dict_by_keys(norm_feats)
        self._norm_sample_weights[task_name] = norm_ws
        self._norm_sample_feats[task_name] = norm_feats

    def _build_designer_kernel(self, beta):
        def likelihood_fn(true_w, sample_w, task):
            state = self._get_init_state(task)
            actions = self._controller(state, weights=sample_w)
            _, cost, info = self._runner(state, actions, weights=true_w)
            rew = -1 * cost
            return beta * rew

        return likelihood_fn

    def _build_kernel(self, beta):
        """Likelihood for observed data.

        Args:
            prior_w (dict): sampled from prior
            user_w (dict): user-specified reward weight
            tasks (array): environment init state
            norm_sample_feats(array): samples used to normalize likelihood in
                inverse reward design problem. Sampled before running `pgm.sample`.

        Example:
            >>> log_prob = likelihood_fn(weight, task)

        """

        @jax.jit
        def _likelihood(sample_w, norm_sample_feats, feats):
            """IRD likelihood backbone with JAX speed up.

            Runs `p(w_obs | w)`

            """
            sample_cost = np.sum([feats[key] * sample_w[key] for key in feats.keys()])
            sample_rew = -1 * sample_cost

            normalizing_rews = []
            for key in norm_sample_feats.keys():
                # sum over episode
                w = sample_w[key]
                rew = -1 * np.sum(w * norm_sample_feats[key], axis=1)
                normalizing_rews.append(rew)

            ## Normalizing constant
            N = len(norm_sample_feats)
            log_norm_rew = -np.log(N) + logsumexp(np.sum(normalizing_rews, axis=0))

            log_prob = beta * (sample_rew - log_norm_rew)
            return log_prob

        def likelihood_fn(user_w, sample_w, norm_sample_feats, user_feats):
            """Main likelihood logic.

            """
            log_prob = _likelihood(sample_w, norm_sample_feats, user_feats)
            # print(f"likelihood {log_prob} prior {self._prior_log_prob(sample_w)}")
            log_prob += self._prior_log_prob(sample_w)
            return log_prob

        return likelihood_fn
