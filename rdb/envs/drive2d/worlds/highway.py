import gym
import jax
import jax.numpy as np
from rdb.envs.drive2d.core.world import DriveWorld
from rdb.envs.drive2d.core import lane, feature
from rdb.optim.utils import *


class HighwayDriveWorld(DriveWorld):
    """Straight multi-lane environment.

    For modeling lane changes

    Attributes:
        main_car: main controlled vehicle

    Methods
        build_lanes: build lanes given number & width
        build_fences: build fences around lanes
        get_feat_fns: dict of feature functions
                      e.g. feat_fns['main_car'](state) = main_car_s

    """

    def __init__(
        self, main_car, cars=[], num_lanes=3, objects=[], lane_width=0.13, dt=0.1
    ):
        self._lanes = self.build_lanes(num_lanes, lane_width)
        self._fences = self.build_fences(self._lanes)
        self._lane_width = lane_width
        super().__init__(main_car, cars, self._lanes, dt, objects=objects)

    @property
    def fences(self):
        return self._fences

    @property
    def lane_width(self):
        return self._lane_width

    def build_lanes(self, num_lanes, lane_width):
        min_shift = -(num_lanes - 1) / 2.0
        max_shift = (num_lanes - 1) / 2.0 + 0.001
        lane_shifts = np.arange(min_shift, max_shift, 1.0)
        clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], lane_width)
        # clane = lane.StraightLane([0.0, -.1], [0.0, .1], lane_width)
        lanes = [clane.shifted(s) for s in lane_shifts]
        return lanes

    def build_fences(self, lanes):
        # fences = [lanes[0].shifted(-0.5), lanes[-1].shifted(0.5)]
        fences = [lanes[0].shifted(-1), lanes[-1].shifted(1)]
        return fences

    def _get_raw_features_dict(self):
        feats_dict = super()._get_raw_features_dict()

        ## Distance to fences
        fence_fns = [None] * len(self._fences)
        normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
        for f_i, (fence, normal) in enumerate(zip(self._fences, normals)):
            main_idx = self._indices["main_car"]

            def fence_dist_fn(state, actions, fence=fence, normal=normal):
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist_outside_fence(main_pos, fence.center, normal)
                # return feature.diff_to_fence(fence.center, normal, main_pos)

            fence_fns[f_i] = fence_dist_fn
        feats_dict["dist_fences"] = concat_funcs(fence_fns, axis=0)
        feats_dict["dist_fences"] = jax.jit(feats_dict["dist_fences"])

        return feats_dict
