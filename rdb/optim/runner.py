import numpy as np
from functools import partial


"""
Forward environments and collect trajectorys
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

    def __call__(self, x0, u, weights=None):
        """
        Param
        : x0 :
        : u  : array(T, u_dim), actions
        """
        # TODO: action space shape checking
        length = len(u)
        cost_fn = self._cost_fn
        if weights is not None:
            cost_fn = partial(self._cost_runtime, weights=weights)

        x = x0
        xs = [x]
        total_cost = 0.0
        info = dict(costs=[])

        for t in range(length):
            next_x = x + self._dynamics_fn(x, u[t]) * self._dt
            cost = cost_fn(x, u[t])
            total_cost += cost
            info["costs"].append(x)
            xs.append(next_x)
            x = next_x
        return np.array(xs), cost, info
