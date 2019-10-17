import abc
import gym
import jax.numpy as np
from rdb.envs.drive2d.core.world import DriveWorld
from rdb.envs.drive2d.core import lane, feature
from rdb.optim.utils import *


class HighwayDriveWorld(DriveWorld):
    def __init__(self, main_car, cars=[], num_lanes=3, lane_width=0.13, dt=0.1):
        lanes, fences = self.build_lanes(num_lanes, lane_width)
        self._fences = self.build_fences(lanes)
        super().__init__(main_car, cars, lanes, dt)

    @property
    def fences(self):
        return self._fences

    def build_lanes(self, num_lanes, lane_width):
        min_shift = -(num_lanes - 1) / 2.0
        max_shift = (num_lanes - 1) / 2.0 + 0.001
        lane_shifts = np.arange(min_shift, max_shift, 1.0)
        clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], lane_width)
        lanes = [clane.shifted(s) for s in lane_shifts]
        fences = [clane.shifted(min_shift - 1), clane.shifted(max_shift + 1)]
        return lanes, fences

    def build_fences(self, lanes):
        fences = [lanes[0].shifted(-1), lanes[-1].shifted(1)]
        return fences

    def get_feat_fns(self, indices):
        fns = super().get_feat_fns(indices)

        fence_fns = [None] * len(self._fences)
        for f_i, fence in enumerate(self._fences):
            main_idx = indices["main_car"]

            def fence_dist_fn(state, actions, fence=fence):
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist2lane(fence.center, fence.normal, main_pos)

            fence_fns[f_i] = fence_dist_fn
        fns["dist_fences"] = concat_funcs(fence_fns)

        return fns
