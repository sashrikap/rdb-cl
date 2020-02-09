import os
import rdb
import time
import numpy as np
from imageio import imsave
from functools import partial
from rdb.infer import *
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
        roll_forward (fn): return (T, nbatch, xdim)
        roll_costs (fn): return (T, nbatch, 1)

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
        # JIT compile
        self._a_shape = None

    @property
    def env(self):
        return self._env

    def run_from_ipython(self):
        try:
            __IPYTHON__
            return True
        except NameError:
            return False

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
        """Save mp4 data.

        Args:
            state(ndarray): (nbatch, xdim)
            actions(ndarray): (T, nbatch, udim)

        """
        assert len(state.shape) == 2
        assert len(actions.shape) == 3
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "recording.mp4")
        os.makedirs(dirname(path), exist_ok=True)
        self._env.reset()
        self._env.state = state
        render_env(self._env, state, actions, fps=3, path=path, text=text)
        self._env.close_window()
        return path

    def nb_show_mp4(self, state, actions, path, clear=True):
        """ Visualize mp4 on Jupyter notebook

        Args:
            state(ndarray): (nbatch, xdim)
            actions(ndarray): (T, nbatch, udim)

        """
        from ipywidgets import Output
        from IPython.display import display, Image, Video, clear_output

        assert len(state.shape) == 2
        assert len(actions.shape) == 3
        if os.path.isfile(path):
            os.remove(path)
        FRAME_WIDTH = 450
        mp4_path = self.collect_mp4(state, actions, path=path, width=FRAME_WIDTH)
        os.makedirs(dirname(mp4_path), exist_ok=True)
        if clear:
            clear_output()
        print(f"video path {mp4_path}")
        display(Video(mp4_path, width=FRAME_WIDTH))

    def collect_thumbnail(self, state, width=450, path=None, text=None):
        """Save thumbnail data.

        Args:
            state(ndarray): (nbatch, xdim)

        """
        assert len(state.shape) == 2
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "thumbnail.png")
        os.makedirs(dirname(path), exist_ok=True)
        self._env.reset()
        self._env.state = state
        frame = self._env.render("rgb_array", text=text)
        frame = imresize(frame, (width, width))
        if self.run_from_ipython():
            self._env.close_window()
        imsave(path, frame)

    def collect_heatmap(self, state, weights, width=450, path=None, text=None):
        assert len(state.shape) == 2

        weights = DictList(weights)
        if path is None:
            path = join(dirname(rdb.__file__), "..", "data", "thumbnail.png")
        os.makedirs(dirname(path), exist_ok=True)
        self._env.reset()
        self._env.state = state
        frame = self._env.render(
            "rgb_array", draw_heat=True, weights=weights, text=text
        )
        frame = imresize(frame, (width, width))
        if self.run_from_ipython():
            self._env.close_window()
        imsave(path, frame)

    def nb_show_thumbnail(self, state, path, clear=True):
        """ Visualize mp4 on Jupyter notebook

        Args:
            state(ndarray): (nbatch, xdim)

        """

        from ipywidgets import Output
        from IPython.display import display, Image, clear_output

        assert len(state.shape) == 2
        if os.path.isfile(path):
            os.remove(path)
        FRAME_WIDTH = 450
        self.collect_thumbnail(state, path=path, width=FRAME_WIDTH)
        if clear:
            clear_output()
        # display(Video(mp4_path, width=FRAME_WIDTH))
        display(Image(path))

    def __call__(self, x0, actions, weights, batch=True):
        """Run optimization.

        Args:
            x0 (ndarray): initial states
                shape (nbatch, xdim)
            actions (ndarray): actions
                shape (nbatch, T, udim)
            batch (bool), batch mode. If `true`, weights are batched
                If `false`, weights are not batched

        Return:
            xs (ndarray): all trajectory states
                shape (nbatch, T, xdim)
            cost_sum: (nbatch, )
            info (dict): rollout info

            info['costs'] (DictList): nfeats * (nbatch, T)
            info['feats'] (DictList): nfeats * (nbatch, T)
            info['feats_sum'] (DictList): nfeats * (nbatch, )
            info['violations'] (DictList): ncons * (nbatch, T)
            info['vios_sum'] (DictList): ncons * (nbatch,)
            info['metadata'] (DictList): nmeta * (nbatch, T)

        """
        assert self._roll_costs is not None, "Cost function improperly defined"

        weights_arr = (
            DictList(weights, expand_dims=not batch)
            .prepare(self._env.features_keys)
            .numpy_array()
        )

        # Track JIT recompile
        t_compile = None
        a_shape = actions.shape
        if self._a_shape is None:
            print(f"JIT - Runner first compile: ac {a_shape}")
            self._a_shape = a_shape
            t_compile = time()
        elif actions.shape != self._a_shape:
            print(
                f"JIT - Runner recompile: ac {actions.shape}, previously {self._a_shape}"
            )
            self._a_shape = a_shape
            t_compile = time()

        # Rollout
        #  shape (nbatch, T, udim) -> (T, nbatch, udim)
        actions = actions.swapaxes(0, 1)
        #  shape (T, nbatch, xdim)
        xs = self._roll_forward(x0, actions)
        #  shape (T, nbatch,)
        costs = self._roll_costs(x0, actions, weights_arr)
        #  shape (nbatch, T,)
        costs = costs.swapaxes(0, 1)
        #  shape nfeats * (T, nbatch)
        feats = DictList(self._roll_features(x0, actions))
        #  shape nfeats * (nbatch, T)
        feats = feats.transpose()

        # Track JIT recompile
        if t_compile is not None:
            print(
                f"JIT - Runner finish compile in {time() - t_compile:.3f}s: ac {self._a_shape}"
            )

        # Compute features
        #  shape (nbatch, )
        cost_sum = costs.sum(axis=1)
        #  shape nfeats * (nbatch,)
        feats_sum = feats.sum(axis=1)
        #  shape ncons * (T, nbatch,)
        violations = DictList(self._env.constraints_fn(xs, actions))
        violations = violations.transpose()
        #  shape ncons * (nbatch, T,)
        vios_sum = violations.sum(axis=1)
        #  shape nneta * (T, nbatch,)
        metadata = DictList(self._env.metadata_fn(xs, actions))
        metadata = metadata.transpose()

        # DictList conveniently deals with list of dicts
        info = {}
        info["costs"] = costs
        info["cost_sum"] = cost_sum
        info["feats"] = feats
        info["feats_sum"] = feats_sum
        info["violations"] = violations
        info["vios_sum"] = vios_sum
        info["metadata"] = metadata

        return onp.array(xs), cost_sum, info
