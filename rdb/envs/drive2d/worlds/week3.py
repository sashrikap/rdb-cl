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
        car_length=0.1,
        car_width=0.08,
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, np.array(state), speed))
        main_car = OptimalControlCar(self, weights, main_state, horizon)
        self.goal_speed = goal_speed
        self.goal_lane = goal_lane
        self.control_bound = control_bound
        self._car_length = car_length
        self._car_width = car_width
        super().__init__(main_car, cars, num_lanes=num_lanes, dt=dt)
        self.cost_features = self.build_cost_features()

    def build_cost_features(self):
        """
        Types:

        : Gaussian : exp(-dist^2/sigma^2)
        : Exponent : exp(run_over)
        """
        features = {}
        sum_keep = partial(np.sum, keepdims=True)
        # Gaussian
        features["dist_cars"] = compose(
            np.sum,
            partial(
                gaussian_feat,
                # sigma=self._car_length,
                sigma=np.array([self._car_width, self._car_length]),
                # mu=2 * np.array([self._car_width, self._car_length]),
            ),
        )
        # Gaussian
        features["dist_lanes"] = compose(
            np.sum,
            neg_feat,
            partial(gaussian_feat, sigma=self._car_length),
            partial(index_feat, index=self.goal_lane),
        )
        # Quadratic barrier function
        """features["dist_fences"] = compose(
            np.sum, partial(sigmoid_feat, mu=8.0 / self._car_width), neg_feat
        )"""
        features["dist_fences"] = compose(
            np.sum,
            quadratic_feat,
            neg_relu_feat,
            partial(diff_feat, subtract=self._car_width),
        )
        bound = self.control_bound
        # features["control"] = partial(bounded_feat, lower=-bound, upper=bound, width=0.5)
        features["control"] = compose(np.sum, quadratic_feat)
        features["speed"] = compose(
            np.sum, partial(quadratic_feat, goal=self.goal_speed)
        )
        return features


class Week3_01(HighwayDriveWorld_Week3):
    def __init__(self):
        main_speed = 0.8
        main_state = np.array([0, 0, np.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.2
        control_bound = 0.5
        ## Weird 1
        # car1 = np.array([0.0, 0.45, np.pi / 2, 0])
        # car2 = np.array([-0.125, 0.4, np.pi / 2, 0])
        ## Weird 2
        # car1 = np.array([0.0, 0.45, np.pi / 2, 0])
        # car2 = np.array([-0.125, 0.4, np.pi / 2, 0])
        car1 = np.array([0.0, 0.3, np.pi / 2, 0])
        car2 = np.array([-0.125, 0.9, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([0.6, 0.6])
        weights_191023 = {
            "dist_cars": 100.0,
            "dist_lanes": 10.0,
            "dist_fences": 200.0,
            # "dist_fences": 5,
            "speed": 4.0,
            "control": 80.0,
        }
        weights = {
            "dist_cars": 100.0,
            "dist_lanes": 100.0,
            "dist_fences": 200.0,
            # "dist_fences": 5,
            "speed": 16.0,
            "control": 80.0,
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
