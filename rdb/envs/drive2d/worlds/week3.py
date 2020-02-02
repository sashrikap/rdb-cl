"""
One autonomous car is merging with two other fix speed cars
"""


import jax
import jax.numpy as np
import numpyro
import numpyro.distributions as dist
import itertools, copy
from collections import OrderedDict
from rdb.optim.utils import *
from rdb.envs.drive2d.core import car, objects, feature, constraints
from rdb.envs.drive2d.worlds.highway import HighwayDriveWorld
from functools import partial
from numpyro.handlers import seed
from rdb.infer.utils import random_choice


class HighwayDriveWorld_Week3(HighwayDriveWorld):
    def __init__(
        self,
        main_state,
        goal_speed,
        goal_lane,
        control_bound,
        car_states=[],
        car_speeds=[],
        dt=0.1,
        horizon=10,
        num_lanes=3,
        lane_width=0.13,
        car1_range=[-0.8, 0.8],
        car2_range=[-0.8, 0.8],
        car_delta=0.2,
        obstacle_states=[],
    ):
        # Define cars
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(car.FixSpeedCar(self, np.array(state), speed))
        main_car = car.OptimalControlCar(self, main_state, horizon)
        self._goal_speed = goal_speed
        self._goal_lane = goal_lane
        self._control_bound = control_bound
        # Define objects
        objs = []
        for state in obstacle_states:
            objs.append(objects.Obstacle(np.array(state)))
        super().__init__(main_car, cars, num_lanes=num_lanes, objects=objs, dt=dt)
        # Define all tasks to sample from
        self._car1_range = np.arange(car1_range[0], car1_range[1], car_delta)
        self._car2_range = np.arange(car2_range[0], car2_range[1], car_delta)
        self._grid_tasks = (self._car1_range, self._car2_range)
        self._all_tasks = list(itertools.product(self._car1_range, self._car2_range))

    def set_task(self, task):
        assert len(task) == 2
        y0_idx, y1_idx = 1, 5
        self.reset()
        state = copy.deepcopy(self.state)
        state[y0_idx] = task[0]
        state[y1_idx] = task[1]
        self.set_init_state(state)

    def _get_nonlinear_features_dict(self, feats_dict):
        """
        Args:
            feats_dict (list): dict of environment feature functions

        Note:
            * Gaussian : exp(-dist^2/sigma^2)
            * Exponent : exp(run_over)

        """
        nlr_feats_dict = {}
        sum_keep = partial(np.sum, keepdims=True)
        # Gaussian
        ncars = len(self._cars)
        nlr_feats_dict["dist_cars"] = compose(
            np.sum,
            partial(
                feature.gaussian_feat,
                sigma=np.array([self._car_width / 2, self._car_length]),
            ),
        )
        # Gaussian
        nlr_feats_dict["dist_lanes"] = compose(
            np.sum,
            feature.neg_feat,
            partial(feature.gaussian_feat, sigma=self._car_length),
            partial(feature.item_index_feat, index=self._goal_lane),
        )
        """nlr_feats_dict["dist_lanes"] = compose(
            np.sum, quadratic_feat, partial(item_index_feat, index=self._goal_lane)
        )"""
        # Quadratic barrier function
        """nlr_feats_dict["dist_fences"] = compose(
            np.sum, partial(sigmoid_feat, mu=8.0 / self._car_width), neg_feat
        )"""
        nlr_feats_dict["dist_fences"] = compose(
            np.sum,
            feature.quadratic_feat,
            feature.neg_relu_feat,
            lambda dist: dist - (self._lane_width + self._car_length) / 2,
        )
        """nlr_feats_dict["dist_fences"] = compose(
            np.sum,
            partial(
                feature.gaussian_feat,
                sigma=np.array([self._car_width / 3, self._car_length / 3]),
            ),
            # debug_print
        )"""
        bound = self._control_bound
        nlr_feats_dict["control"] = compose(np.sum, feature.quadratic_feat)
        nlr_feats_dict["speed"] = compose(
            np.sum, partial(feature.quadratic_feat, goal=self._goal_speed)
        )

        nlr_feats_dict = chain_dict_funcs(nlr_feats_dict, feats_dict)
        # Speed up
        for key, fn in nlr_feats_dict.items():
            nlr_feats_dict[key] = jax.jit(fn)

        return nlr_feats_dict

    def _get_constraints_fn(self):
        constraints_dict = OrderedDict()
        constraints_dict["offtrack"] = constraints.build_offtrack(env=self)
        constraints_dict["overspeed"] = constraints.build_overspeed(
            env=self, max_speed=1.0
        )
        constraints_dict["underspeed"] = constraints.build_underspeed(
            env=self, min_speed=0.2
        )
        constraints_dict["uncomfortable"] = constraints.build_uncomfortable(
            env=self, max_actions=self._control_bound
        )
        constraints_dict["wronglane"] = constraints.build_wronglane(
            env=self, lane_idx=2
        )
        constraints_dict["collision"] = constraints.build_collision(env=self)
        constraints_fn = merge_dict_funcs(constraints_dict)

        return constraints_dict, constraints_fn

    def _get_metadata_fn(self):
        metadata_dict = OrderedDict()
        metadata_dict["overtake0"] = constraints.build_overtake(env=self, car_idx=0)
        metadata_dict["overtake1"] = constraints.build_overtake(env=self, car_idx=1)
        metadata_fn = merge_dict_funcs(metadata_dict)
        return metadata_dict, metadata_fn

    def update_key(self, rng_key):
        super().update_key(rng_key)


class Week3_01(HighwayDriveWorld_Week3):
    def __init__(self):
        main_speed = 0.8
        car_speed = 0.6
        main_state = np.array([0, 0, np.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        control_bound = [0.5, 1.0]
        lane_width = 0.13
        num_lanes = 3
        ## Weird 1
        # car1 = np.array([0.0, 0.45, np.pi / 2, 0])
        # car2 = np.array([-0.125, 0.4, np.pi / 2, 0])
        ## Weird 2
        # car1 = np.array([0.0, 0.45, np.pi / 2, 0])
        # car2 = np.array([-0.125, 0.4, np.pi / 2, 0])
        car1 = np.array([0.0, 0.3, np.pi / 2, 0])
        car2 = np.array([-lane_width, 0.9, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([car_speed, car_speed])
        super().__init__(
            main_state,
            goal_speed=goal_speed,
            goal_lane=goal_lane,
            control_bound=control_bound,
            car_states=car_states,
            car_speeds=car_speeds,
            dt=dt,
            horizon=horizon,
            num_lanes=num_lanes,
            lane_width=lane_width,
        )


class Week3_02(HighwayDriveWorld_Week3):
    def __init__(self):
        main_speed = 0.7
        car_speed = 0.5
        main_state = np.array([0, 0, np.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        control_bound = 0.5
        lane_width = 0.13
        num_lanes = 3
        car1 = np.array([0.0, 0.3, np.pi / 2, 0])
        car2 = np.array([-lane_width, 0.9, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([car_speed, car_speed])

        super().__init__(
            main_state,
            goal_speed=goal_speed,
            goal_lane=goal_lane,
            control_bound=control_bound,
            car_states=car_states,
            car_speeds=car_speeds,
            dt=dt,
            horizon=horizon,
            num_lanes=num_lanes,
            lane_width=lane_width,
        )
