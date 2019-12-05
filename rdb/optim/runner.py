import numpy as np
import time
import rdb
from os.path import join, dirname
from functools import partial
from collections import Counter, OrderedDict
from scipy.misc import imresize
from imageio import imsave
from rdb.visualize.render import forward_env, render_env

"""
Forward environments and collect trajectorys

TODO:
[1] shape checking
"""


class Runner(object):
    """Basic Runner, collects trajectory, features and cost.

    Args:
        env (object): environment
        dynamcis_fn (fn): can pass in jit-accelerated function
        cost_runtime (fn): can pass in jit-accelerated function

    Examples:
        >>> xs = dynamics_fn(x0, us)
        >>> costs = cost_runtime(x0, us, weights)

    """

    def __init__(self, env, dynamics_fn=None, cost_runtime=None):
        self._env = env
        self._dt = env.dt
        # Dynamics function
        self._dynamics_fn = dynamics_fn
        if dynamics_fn is None:
            self.dynamics_fn = env.dynamics_fn
        # Define cost function
        self._cost_runtime = cost_runtime

    @property
    def env(self):
        return self._env

    def _collect_features(self, xs, actions):
        """
        Return
        : feats     : dict(key, [f_t0, f_t1, ..., f_tn]), time series
        : feats_sum : dict(key, sum([f_t0, f_t1, ..., f_tn])), sum
        """
        feats = None
        for x, act in zip(xs, actions):
            feats_x = self._env.features_fn(x, act)
            if feats is None:
                feats = OrderedDict()
                for key in feats_x:
                    feats[key] = []
            for key in feats_x:
                feats[key].append(feats_x[key])
        # Sum up each feature
        feats = OrderedDict({key: np.array(val) for (key, val) in feats.items()})
        feats_sum = OrderedDict(
            {key: np.sum(val, axis=0) for (key, val) in feats.items()}
        )
        return feats, feats_sum

    def collect_frames(self, actions, width=450, mode="rgb_array", text=""):
        self._env.reset()
        frames = []
        if mode == "rgb_array":
            frames = [self._env.render("rgb_array", text=text)]
        # Rollout
        for act in actions:
            self._env.step(act)
            if mode == "human":
                self._env.render("human")
                time.sleep(0.1)
            elif mode == "rgb_array":
                frame = self._env.render("rgb_array", text=text)
                frame = imresize(frame, (width, width))
                frames.append(frame)
            else:
                raise NotImplementedError
        return frames

    def collect_mp4(self, state, actions, width=450, path=None, text=None):
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "recording.mp4")
        self._env.reset()
        self._env.state = state
        render_env(self._env, state, actions, fps=3, path=path, text=text)
        return path

    def collect_thumbnail(self, state, actions, width=450, path=None, text=None):
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "thumbnail.png")
        self._env.reset()
        self._env.state = state
        frame = self._env.render("rgb_array", text=text)
        frame = imresize(frame, (width, width))
        imsave(path, frame)

    def __call__(self, x0, actions, weights=None, collect_features=False):
        """Run optimization.

        Args:
            x0 (ndarray(xdim)):
            actions array(T, u_dim): actions

        """
        # TODO: action space shape checking
        assert self._cost_runtime is not None, "Cost function improperly defined"
        cost_fn = partial(self._cost_runtime, weights=weights)

        x = x0
        info = dict(costs=[], feats={}, feats_sum={})
        xs = self._dynamics_fn(x0, actions)
        info["costs"] = self._cost_runtime(x0, actions, weights)
        cost = np.sum(info["costs"])
        if collect_features:
            info["feats"], info["feats_sum"] = self._collect_features(xs, actions)
        return np.array(xs), cost, info
