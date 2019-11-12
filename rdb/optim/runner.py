import numpy as np
import time
from functools import partial
from collections import Counter, OrderedDict
from scipy.misc import imsave, imresize
from rdb.visualize.render import forward_env

"""
Forward environments and collect trajectorys

TODO:
[1] shape checking
"""


class Runner(object):
    """
    Basic Runner, collects
    : xs    : raw trajectory
    : feats : features
    """

    def __init__(self, env, agent):
        self._env = env
        self._agent = agent
        self._dt = env.dt
        self._dynamics_fn = env.dynamics_fn
        self._cost_runtime = agent.cost_runtime
        self._cost_fn = agent.cost_fn

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
            feats_x = self.env.features_fn(x, act)
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

    def __call__(self, x0, actions, weights=None):
        """
        Param
        : x0      :
        : actions : array(T, u_dim), actions
        """
        # TODO: action space shape checking
        length = len(actions)
        cost_fn = self._cost_fn
        if weights is not None:
            cost_fn = partial(self._cost_runtime, weights=weights)

        x = x0
        xs = [x]
        total_cost = 0.0
        info = dict(costs=[], feats={}, feats_sum={})

        for t in range(length):
            next_x = x + self._dynamics_fn(x, actions[t]) * self._dt
            cost = cost_fn(x, actions[t])
            total_cost += cost
            info["costs"].append(x)
            xs.append(next_x)
            x = next_x

        info["feats"], info["feats_sum"] = self._collect_features(xs, actions)
        return np.array(xs), cost, info
