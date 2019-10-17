import jax.numpy as np
from rdb.envs.drive2d.worlds.highway import HighwayDriveWorld
from rdb.envs.drive2d.core.car import *
from rdb.envs.drive2d.core.feature import *
from functools import partial
from toolz.functoolz import compose

"""
Sample adversarial scenarios where one autonomous car is merging
with two other fix speed cars
"""


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
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, np.array(state), speed))
        main_car = OptimalControlCar(self, weights, main_state, horizon)
        self.goal_speed = goal_speed
        self.goal_lane = goal_lane
        self.control_bound = control_bound
        super().__init__(main_car, cars, num_lanes=num_lanes, dt=dt)
        self.rew_features = self.build_rew_features()

    def build_rew_features(self):
        features = {}
        sum_keep = partial(np.sum, keepdims=True)
        features["dist_cars"] = compose(sum_keep, partial(exp_feat, sigma=0.3))
        features["dist_lanes"] = compose(
            quadratic_feat, partial(index_feat, index=self.goal_lane)
        )
        features["dist_fences"] = compose(sum_keep, partial(exp_feat, sigma=0.3))

        bound = self.control_bound
        # features["control"] = partial(bounded_feat, lower=-bound, upper=bound, width=0.5)
        features["control"] = quadratic_feat
        features["speed"] = partial(abs_feat, goal=self.goal_speed)
        return features


class Week3_01(HighwayDriveWorld_Week3):
    def __init__(self):
        main_speed = 0.8
        main_state = np.array([0, 0, np.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 20
        dt = 0.1
        control_bound = 0.5
        car1 = np.array([0.0, 0.2, np.pi / 2, 0])
        car2 = np.array([-0.125, 0.2, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([0.6, 0.6])
        weights = {
            "dist_cars": -1.0,
            "dist_lanes": -5.0,
            "dist_fences": -0.5,
            "speed": -2.0,
            "control": -1.0,
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
        )
