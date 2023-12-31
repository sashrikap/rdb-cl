"""
Drive into a parking area, with other cars passing.
"""


import jax
import jax.numpy as jnp
from rdb.optim.utils import *
from rdb.envs.drive2d.core.car import *
from rdb.envs.drive2d.core.constraints import *
from rdb.envs.drive2d.core.feature import *
from rdb.envs.drive2d.worlds.driveway import *
from functools import partial


class EntranceDriveWorld_Week4(EntranceDriveWorld):
    def __init__(
        self,
        main_state,
        goal_speed,
        goal_lane,
        control_bound,
        car_states=[],
        car_speeds=[],
        driveway_dist=0.8,
        dt=0.1,
        horizon=10,
        lane_width=0.13,
    ):
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(FixSpeedCar(self, jnp.array(state), speed))
        main_car = OptimalControlCar(self, main_state, horizon)
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
        nlr_feats_dict = OrderedDict()
        sum_keep = partial(jnp.sum, keepdims=True)
        # Gaussian
        ncars = len(self._cars)
        nlr_feats_dict["dist_cars"] = compose(
            jnp.sum,
            partial(
                gaussian_feat, sigma=jnp.array([self._car_width / 2, self._car_length])
            ),
        )
        # Gaussian
        nlr_feats_dict["dist_lanes"] = compose(
            jnp.sum,
            neg_feat,
            partial(gaussian_feat, sigma=self._car_length),
            partial(item_index_feat, index=self.goal_lane),
        )
        nlr_feats_dict["dist_fences"] = compose(
            jnp.sum,
            quadratic_feat,
            neg_relu_feat,
            lambda dist: dist - (self._lane_width + self._car_length) / 2,
        )
        bound = self._control_bound
        nlr_feats_dict["control"] = compose(jnp.sum, quadratic_feat)
        nlr_feats_dict["speed"] = compose(
            jnp.sum, partial(quadratic_feat, goal=self.goal_speed)
        )
        nlr_feats_dict["dist_entrance"] = compose(jnp.sum, quadratic_feat)
        nlr_feats_dict["dist_garage"] = compose(jnp.sum, quadratic_feat)
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
        main_state = jnp.array([lane_width / 2, 0, jnp.pi / 2, main_speed])
        driveway_dist = 0.8
        car1 = jnp.array([-lane_width / 2, 0.3, jnp.pi / 2, 0])
        car2 = jnp.array([-lane_width / 2, -0.3, jnp.pi / 2, 0])
        car_states = jnp.array([car1, car2])
        car_speeds = jnp.array([car_speed, car_speed])
        super().__init__(
            main_state,
            goal_speed,
            goal_lane,
            control_bound,
            car_states,
            car_speeds,
            driveway_dist,
            dt,
            horizon,
            lane_width,
        )
