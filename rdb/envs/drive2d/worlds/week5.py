import jax
import jax.numpy as np
from rdb.optim.utils import *
from rdb.envs.drive2d.core.car import *
from rdb.envs.drive2d.core.feature import *
from rdb.envs.drive2d.worlds.highway import HighwayDriveWorld
from functools import partial

"""
Sample adversarial scenarios where one autonomous car is merging
with two other fix speed cars
"""


class HighwayDriveWorld_Week5(HighwayDriveWorld):
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
        num_lanes=4,
        lane_width=0.13,
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, np.array(state), speed))
        main_car = OptimalControlCar(self, weights, main_state, horizon)
        self.goal_speed = goal_speed
        self.goal_lane = goal_lane
        self.control_bound = control_bound
        super().__init__(main_car, cars, num_lanes=num_lanes, dt=dt)

    def set_init_state(self, y0, y1):
        """
        Setting initial state
        """
        # y0_idx, y1_idx = 1, 5
        self.cars[0].init_state = jax.ops.index_update(self.cars[0].init_state, 1, y0)
        self.cars[1].init_state = jax.ops.index_update(self.cars[1].init_state, 1, y1)

    def _get_nonlinear_features_dict(self, feats_dict):
        """
        Params
        : feats_dict : dict of environment feature functions
        Types
        : Gaussian   : exp(-dist^2/sigma^2)
        : Exponent   : exp(run_over)
        """
        nonlinear_dict = {}
        sum_keep = partial(np.sum, keepdims=True)
        # Gaussian
        nonlinear_dict["dist_cars"] = compose(
            np.sum,
            partial(
                gaussian_feat, sigma=np.array([self._car_width / 2, self._car_length])
            ),
        )
        # Gaussian
        nonlinear_dict["dist_lanes"] = compose(
            np.sum,
            neg_feat,
            partial(gaussian_feat, sigma=self._car_length),
            partial(index_feat, index=self.goal_lane),
        )
        nonlinear_dict["dist_fences"] = compose(
            np.sum,
            quadratic_feat,
            neg_relu_feat,
            lambda dist: dist - (self._lane_width + self._car_length) / 2,
        )
        bound = self.control_bound
        nonlinear_dict["control"] = compose(np.sum, quadratic_feat)
        nonlinear_dict["speed"] = compose(
            np.sum, partial(quadratic_feat, goal=self.goal_speed)
        )

        # Speed up
        for key, fn in nonlinear_dict.items():
            nonlinear_dict[key] = jax.jit(fn)

        feats_dict = chain_funcs(nonlinear_dict, feats_dict)
        return feats_dict

    def _get_constraints_fn(self):
        constraints_dict = {}
        constraints_fn = merge_dict_funcs(constraints_dict)

        return constraints_dict, constraints_fn


class Week5_01(HighwayDriveWorld_Week5):
    def __init__(self):
        main_speed = 0.8
        car_speed = 0.6
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        control_bound = 0.5
        lane_width = 0.13
        num_lanes = 4
        main_state = np.array([0.5 * lane_width, 0, np.pi / 2, main_speed])
        car1 = np.array([-0.5 * lane_width, 0.3, np.pi / 2, 0])
        car2 = np.array([-1.5 * lane_width, 0.9, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([car_speed, car_speed])
        weights = {
            "dist_cars": 50,
            "dist_lanes": 30.0,
            "dist_fences": 5000.0,
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