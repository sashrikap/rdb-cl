import gym
import jax
import jax.numpy as np
from rdb.envs.drive2d.core.world import DriveWorld
from rdb.envs.drive2d.core import lane, feature, objects
from rdb.optim.utils import *


class EntranceDriveWorld(DriveWorld):
    """Entrance environment.

    For modeling leaving/entering driveway.

    Attributes:
        main_car: main controlled vehicle

    Methods:
        build_lanes: see rdb/envs/drive2d/worlds/highway.py
        build_fences: see rdb/envs/drive2d/worlds/highway.py
        get_feat_fns: see rdb/envs/drive2d/worlds/highway.py

    """

    NUM_LANES = 2

    def __init__(
        self,
        main_car,
        cars=[],
        lane_width=0.13,
        driveway_width=0.35,
        driveway_length=0.5,
        driveway_dist=0.8,
        dt=0.1,
    ):
        self._lanes = self.build_lanes(lane_width)
        self._fences = self.build_fences(self._lanes)
        self._lane_width = lane_width
        driveway_state = [-self._lane_width, driveway_dist, np.pi / 2, 0]
        driveway_pt1 = [0.5, 1.0]
        driveway_pt2 = [-2.5, 1.0]
        # driveway_pt1 = [0., -1.0]
        # driveway_pt1 = [0., .0]
        self._driveway = objects.Entrance(
            driveway_state, driveway_pt1, driveway_pt2, driveway_width, driveway_length
        )

        garage_state = [-self._lane_width - driveway_length, driveway_dist, 0, 0.0]
        # garage_state = [0, 0, 0, 0]
        self._garage = objects.Garage(garage_state, 10.0)
        env_objects = [self._driveway, self._garage]

        super().__init__(main_car, cars, self._lanes, dt, objects=env_objects)

    @property
    def fences(self):
        return self._fences

    @property
    def lane_width(self):
        return self._lane_width

    def build_lanes(self, lane_width):
        min_shift = -(self.NUM_LANES - 1) / 2.0
        max_shift = (self.NUM_LANES - 1) / 2.0 + 0.001
        lane_shifts = np.arange(min_shift, max_shift, 1.0)
        clane = lane.StraightLane([0.0, -1.0], [0.0, 1.0], lane_width)
        lanes = [clane.shifted(s) for s in lane_shifts]
        return lanes

    def build_fences(self, lanes):
        fences = [lanes[0].shifted(-1), lanes[-1].shifted(1)]
        return fences

    def get_raw_features_dict(self):
        feats_dict = super().get_raw_features_dict()
        main_idx = self._indices["main_car"]
        entrance = None
        garage = None
        for o in self._objects:
            if o.name == "Entrance":
                entrance = o
            if "parking" in o.name:
                garage = o
        assert entrance is not None
        assert garage is not None

        # Fence distance feature
        fence_fns = [None] * len(self._fences)
        normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
        for f_i, (fence, normal) in enumerate(zip(self._fences, normals)):

            def fence_dist_fn(state, actions, fence=fence, normal=normal):
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist_inside_fence(main_pos, fence.center, normal)
                # return feature.diff_to_fence(fence.center, normal, main_pos)

            fence_fns[f_i] = fence_dist_fn
        feats_dict["dist_fences"] = concat_funcs(fence_fns, axis=0)
        feats_dict["dist_fences"] = jax.jit(feats_dict["dist_fences"])

        # Entrance driveway distance feature
        def entrance_dist_fn(state, actions, ent=entrance):
            main_pos = state[..., np.arange(*main_idx)]
            return feature.dist_to_segment(main_pos, ent.start, ent.end)

        feats_dict["dist_entrance"] = jax.jit(entrance_dist_fn)

        # Garage distance feature
        def garage_dist_fn(state, actions, garage=garage):
            main_pos = state[..., np.arange(*main_idx)]
            return feature.dist_to(main_pos, garage.state)

        feats_dict["dist_garage"] = jax.jit(garage_dist_fn)

        return feats_dict

    def get_features_keys(self):
        keys = super().get_features_keys()
        keys.append("dist_entrance")
        keys.append("dist_fences")
        keys.append("dist_garage")
        return keys
