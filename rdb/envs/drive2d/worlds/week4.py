import jax
import jax.numpy as np
from rdb.optim.utils import *
from rdb.envs.drive2d.core.car import *
from rdb.envs.drive2d.core.constraints import *
from rdb.envs.drive2d.core.feature import *
from rdb.envs.drive2d.worlds.driveway import *
from functools import partial

"""
Sample adversarial scenarios where one autonomous car is merging
with two other fix speed cars
"""


class EntranceDriveWorld_Week4(EntranceDriveWorld):
    def __init__(
        self,
        main_state,
        goal_speed,
        goal_lane,
        control_bound,
        weights,
        car_states=[],
        car_speeds=[],
        driveway_dist=0.8,
        dt=0.1,
        horizon=10,
        lane_width=0.13,
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, np.array(state), speed))
        main_car = OptimalControlCar(self, weights, main_state, horizon)
        self.goal_speed = goal_speed
        self.goal_lane = goal_lane
        self._control_bound = control_bound
        super().__init__(main_car, cars, driveway_dist=driveway_dist, dt=dt)

    def set_init_state(self, y0, y1):
        """Setting initial state."""
        # y0_idx, y1_idx = 1, 5
        self.cars[0].init_state = jax.ops.index_update(self.cars[0].init_state, 1, y0)
        self.cars[1].init_state = jax.ops.index_update(self.cars[1].init_state, 1, y1)

    def _get_nonlinear_features_dict(self, feats_dict):
        """Given raw features dict, make nonlinear features.

        Args:
            feats_dict: dict of environment feature functions

        Notes:
            * Gaussian feature: exp(-dist^2/sigma^2)
            * Exponential feature: exp(run_over)

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
        bound = self._control_bound
        nonlinear_dict["control"] = compose(np.sum, quadratic_feat)
        nonlinear_dict["speed"] = compose(
            np.sum, partial(quadratic_feat, goal=self.goal_speed)
        )
        nonlinear_dict["dist_entrance"] = compose(np.sum, quadratic_feat)
        nonlinear_dict["dist_garage"] = compose(np.sum, quadratic_feat)

        # Speed up
        for key, fn in nonlinear_dict.items():
            nonlinear_dict[key] = jax.jit(fn)

        feats_dict = chain_funcs(nonlinear_dict, feats_dict)
        return feats_dict

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


class Week4_01(EntranceDriveWorld_Week4):
    def __init__(self):
        main_speed = 0.8
        car_speed = 0.8
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        control_bound = [0.5, 1.0]
        lane_width = 0.13
        main_state = np.array([lane_width / 2, 0, np.pi / 2, main_speed])
        driveway_dist = 0.8
        car1 = np.array([-lane_width / 2, 0.3, np.pi / 2, 0])
        car2 = np.array([-lane_width / 2, -0.3, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([car_speed, car_speed])
        weights = {
            "dist_cars": 0.0,
            "dist_lanes": 0.0,
            "dist_fences": 0.0,
            "dist_entrance": 10.0,
            "dist_garage": 4.0,
            "speed": 0.0,
            "control": 0.0,
        }
        super().__init__(
            main_state,
            goal_speed,
            goal_lane,
            control_bound,
            weights,
            car_states,
            car_speeds,
            driveway_dist,
            dt,
            horizon,
            lane_width,
        )
