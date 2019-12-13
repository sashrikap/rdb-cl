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
from rdb.envs.drive2d.core.car import *
from rdb.envs.drive2d.core.feature import *
from rdb.envs.drive2d.core.constraints import *
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
        weights,
        car_states=[],
        car_speeds=[],
        dt=0.1,
        horizon=10,
        num_lanes=3,
        lane_width=0.13,
        car1_range=[-0.8, 0.8],
        car2_range=[-0.8, 0.8],
        car_delta=0.2,
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, np.array(state), speed))
        weights = sort_dict_based_on_keys(weights, self.features_keys)
        weights_list = weights.values()
        main_car = OptimalControlCar(self, weights_list, main_state, horizon)
        self._goal_speed = goal_speed
        self._goal_lane = goal_lane
        self._control_bound = control_bound
        super().__init__(main_car, cars, num_lanes=num_lanes, dt=dt)
        # Define all tasks to sample from
        self._car1_range = np.arange(car1_range[0], car1_range[1], car_delta)
        self._car2_range = np.arange(car2_range[0], car2_range[1], car_delta)
        self._grid_tasks = (self._car1_range, self._car2_range)
        self._all_tasks = list(itertools.product(self._car1_range, self._car2_range))
        self._task_sampler = None

    def set_task(self, task):
        y0_idx, y1_idx = 1, 5
        self.reset()
        state = copy.deepcopy(self.state)
        state[y0_idx] = task[0]
        state[y1_idx] = task[1]
        self.set_init_state(state)

    def _get_nonlinear_features_list(self, feats_list):
        """
        Args:
            feats_list (list): dict of environment feature functions

        Note:
            * Gaussian : exp(-dist^2/sigma^2)
            * Exponent : exp(run_over)

        """
        nonlinear_dict = {}
        sum_keep = partial(np.sum, keepdims=True)
        # Gaussian
        nonlinear_dict["dist_cars"] = compose(
            np.sum,
            partial(
                gaussian_feat, sigma=np.array([self._car_width / 2, self._car_length])
            ),
            # debug_printwe
            # abs_feat,
        )
        # Gaussian
        nonlinear_dict["dist_lanes"] = compose(
            np.sum,
            neg_feat,
            partial(gaussian_feat, sigma=self._car_length),
            partial(index_feat, index=self._goal_lane),
        )
        """nonlinear_dict["dist_lanes"] = compose(
            np.sum, quadratic_feat, partial(index_feat, index=self._goal_lane)
        )"""
        # Quadratic barrier function
        """nonlinear_dict["dist_fences"] = compose(
            np.sum, partial(sigmoid_feat, mu=8.0 / self._car_width), neg_feat
        )"""
        nonlinear_dict["dist_fences"] = compose(
            np.sum,
            quadratic_feat,
            neg_relu_feat,
            lambda dist: dist - (self._lane_width + self._car_length) / 2,
        )
        """nonlinear_dict["dist_fences"] = compose(
            np.sum,
            partial(
                gaussian_feat,
                sigma=np.array([self._car_width / 3, self._car_length / 3]),
            ),
            # debug_print
        )"""
        bound = self._control_bound
        nonlinear_dict["control"] = compose(np.sum, quadratic_feat)
        nonlinear_dict["speed"] = compose(
            np.sum, partial(quadratic_feat, goal=self._goal_speed)
        )

        # Speed up
        for key, fn in nonlinear_dict.items():
            nonlinear_dict[key] = jax.jit(fn)

        nonlinear_list = list(
            sort_dict_based_on_keys(nonlinear_dict, self.features_keys).values()
        )
        feats_list = chain_list_funcs(nonlinear_list, feats_list)
        return feats_list

    def _get_constraints_fn(self):
        constraints_dict = {}

        constraints_dict["offtrack"] = partial(is_offtrack, env=self)
        constraints_dict["overspeed"] = partial(is_overspeed, env=self, max_speed=1.0)
        constraints_dict["underspeed"] = partial(is_underspeed, env=self, min_speed=0.2)
        constraints_dict["uncomfortable"] = partial(
            is_uncomfortable, env=self, max_actions=self._control_bound
        )
        constraints_dict["wronglane"] = partial(is_wronglane, env=self, lane_idx=2)
        constraints_dict["collision"] = partial(is_collision, env=self)
        constraints_fn = merge_dict_funcs(constraints_dict)

        return constraints_dict, constraints_fn

    def update_key(self, rng_key):
        super().update_key(rng_key)
        self._task_sampler = seed(random_choice, rng_seed=rng_key)


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
        weights_191023 = {
            "dist_cars": 100.0,
            "dist_lanes": 10.0,
            "dist_fences": 200.0,
            "speed": 4.0,
            "control": 80.0,
        }
        weights_191030 = {
            "dist_cars": 1,
            "dist_lanes": 50.0,
            "dist_fences": 1200.0,
            "speed": 500.0,
            "control": 50.0,
        }
        weights_191101 = {
            "dist_cars": 10,
            "dist_lanes": 50.0,
            "dist_fences": 1200.0,
            "speed": 500.0,
            "control": 50.0,
        }
        weights = {
            "dist_cars": 50,
            "dist_lanes": 30.0,
            "dist_fences": 5000.0,
            "speed": 1000.0,
            "control": 20.0,
        }
        # weights = {
        #     "dist_cars": 1,
        #     "dist_lanes": 10.0,
        #     "dist_fences": 1200.0,
        #     "speed": 1000.0,
        #     "control": 50.0,
        # }
        super().__init__(
            main_state,
            goal_speed,
            goal_lane,
            control_bound,
            weights,
            car_states,
            car_speeds,
            dt,
            horizon,
            num_lanes,
            lane_width,
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
        weights = {
            "dist_cars": 0.0,
            "dist_lanes": 0.0,
            "dist_fences": 50.0,
            "speed": 1000.0,
            "control": 20.0,
        }
        super().__init__(
            main_state,
            goal_speed,
            goal_lane,
            control_bound,
            weights,
            car_states,
            car_speeds,
            dt,
            horizon,
            num_lanes,
            lane_width,
        )
