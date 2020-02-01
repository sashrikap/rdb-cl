import os
import rdb
import time
import numpy as np
from imageio import imsave
from functools import partial
from rdb.optim.utils import *
from scipy.misc import imresize
from os.path import join, dirname
from collections import OrderedDict
from rdb.exps.utils import Profiler
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
        roll_forward (fn): can pass in jit-accelerated function
        roll_costs (fn): can pass in jit-accelerated function

    Examples:
        >>> xs = roll_forward(x0, us)
        >>> costs = roll_costs(x0, us, weights)

    """

    def __init__(self, env, roll_forward=None, roll_costs=None, roll_features=None):
        self._env = env
        self._dt = env.dt
        # Dynamics function
        self._roll_forward = roll_forward
        if roll_forward is None:
            self._roll_forward = env.roll_forward
        # Define cost function
        self._roll_costs = roll_costs
        assert roll_costs is not None
        self._roll_features = roll_features
        if roll_features is None:
            self._roll_features = env.roll_features

    @property
    def env(self):
        return self._env

    def run_from_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

    def _collect_features(self, x0, actions):
        """
        Return:
            feats (dict): dict(key, [f_t0, f_t1, ..., f_tn]), time series
            feats_sum (dict): dict(key, sum([f_t0, f_t1, ..., f_tn])), sum

        """
        feats = self._roll_features(x0, actions)
        feats_sum = OrderedDict(
            {key: np.sum(val, axis=0) for (key, val) in feats.items()}
        )
        return feats, feats_sum

    def _collect_metadata(self, xs, actions):
        """Collect extra data, such as overtaking.

        """
        metadata_fn = self._env.metadata_fn
        return metadata_fn(xs, actions)

    def _collect_violations(self, xs, actions):
        """Collect constraint violations from trajectory.

        Args:
            x0 (ndarray): (T, xdim), list of states
            actions (ndarray): (T, udim), list of actions
        Return:
            violations (dict): feats['offtrack']
        Keys:
            `offtrack`, `collision`, `uncomfortable`,
            `overspeed`, `underspeed`, `wronglane`

        """
        constraints_fn = self._env.constraints_fn
        return constraints_fn(xs, actions)

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
        if self.run_from_ipython():
            self._env.close_window()
        return frames

    def collect_mp4(self, state, actions, width=450, path=None, text=None):
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "recording.mp4")
        self._env.reset()
        self._env.state = state
        render_env(self._env, state, actions, fps=3, path=path, text=text)
        self._env.close_window()
        return path

    def nb_show_mp4(self, state, actions, path, clear=True):
        """ Visualize mp4 on Jupyter notebook """
        from ipywidgets import Output
        from IPython.display import display, Image, Video, clear_output

        if os.path.isfile(path):
            os.remove(path)
        FRAME_WIDTH = 450
        mp4_path = self.collect_mp4(state, actions, path=path, width=FRAME_WIDTH)
        if clear:
            clear_output()
        print(f"video path {mp4_path}")
        display(Video(mp4_path, width=FRAME_WIDTH))

    def collect_thumbnail(self, state, actions=None, width=450, path=None, text=None):
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "thumbnail.png")
        self._env.reset()
        self._env.state = state
        frame = self._env.render("rgb_array", text=text)
        frame = imresize(frame, (width, width))
        if self.run_from_ipython():
            self._env.close_window()
        imsave(path, frame)

    def nb_show_thumbnail(self, state, path, clear=True):
        """ Visualize mp4 on Jupyter notebook """
        from ipywidgets import Output
        from IPython.display import display, Image, clear_output

        if os.path.isfile(path):
            os.remove(path)
        FRAME_WIDTH = 450
        self.collect_thumbnail(state, path=path, width=FRAME_WIDTH)
        if clear:
            clear_output()
        # display(Video(mp4_path, width=FRAME_WIDTH))
        display(Image(path))

    def __call__(self, x0, actions, weights=None):
        """Run optimization.

        Args:
            x0 (ndarray(xdim)):
            actions array(T, u_dim): actions

        """
        # TODO: action space shape checking
        assert self._roll_costs is not None, "Cost function improperly defined"
        weights = prepare_weights(weights, self._env.features_keys)
        weights_dict = sort_dict_by_keys(weights, self._env.features_keys)
        weights = np.array(list(weights_dict.values()))

        x = x0
        info = dict(costs=[], feats={}, feats_sum={}, violations={})
        xs = self._roll_forward(x0, actions)

        info["costs"] = self._roll_costs(x0, actions, weights)
        info["feats"], info["feats_sum"] = self._collect_features(x0, actions)
        info["violations"] = self._collect_violations(xs, actions)
        info["metadata"] = self._collect_metadata(xs, actions)

        cost_sum = np.sum(info["costs"])
        return np.array(xs), cost_sum, info
