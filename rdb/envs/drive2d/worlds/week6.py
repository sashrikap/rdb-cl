"""
One autonomous car is merging with two other fix speed cars
There are obstacles (cones) in the highway
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


class HighwayDriveWorld_Week6(HighwayDriveWorld):
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
        obs_range=[-0.16, 0.16, -0.8, 0.8],
        obs_delta=[0.04, 0.2],
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
        self._obs_range_x = np.arange(obs_range[0], obs_range[1], obs_delta[0])
        self._obs_range_y = np.arange(obs_range[2], obs_range[3], obs_delta[1])
        self._grid_tasks = [self._car1_range, self._car2_range]
        for oi in range(len(self._objects)):
            self._grid_tasks.append(self._obs_range_x)
            self._grid_tasks.append(self._obs_range_y)
        self._all_tasks = list(itertools.product(*self._grid_tasks))
        self._task_sampler = None

    def set_task(self, task):
        assert len(task) == 2 + 2 * len(self._objects), "Task format incorrect"
        car_y0_idx, car_y1_idx = 1, 5
        obj_idx = 12
        self.reset()
        state = copy.deepcopy(self.state)
        # Car state
        state[car_y0_idx] = task[0]
        state[car_y1_idx] = task[1]
        # Object state
        state_idx, task_idx = obj_idx, 2
        for obj in self._objects:
            next_task_idx = task_idx + len(obj.state)
            obj.state = np.array(task[task_idx:next_task_idx])
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
                feature.gaussian_feat,
                sigma=np.array([self._car_width / 2, self._car_length]),
            ),
        )
        # Gaussian
        nonlinear_dict["dist_lanes"] = compose(
            np.sum,
            feature.neg_feat,
            partial(feature.gaussian_feat, sigma=self._car_length),
            partial(feature.index_feat, index=self._goal_lane),
        )
        nonlinear_dict["dist_fences"] = compose(
            np.sum,
            feature.quadratic_feat,
            feature.neg_relu_feat,
            lambda dist: dist - (self._lane_width + self._car_length) / 2,
        )
        nonlinear_dict["dist_objects"] = compose(
            np.sum,
            partial(
                feature.gaussian_feat,
                sigma=np.array([self._car_width / 2, self._car_length]),
            ),
        )
        bound = self._control_bound
        nonlinear_dict["control"] = compose(np.sum, feature.quadratic_feat)
        nonlinear_dict["speed"] = compose(
            np.sum, partial(feature.quadratic_feat, goal=self._goal_speed)
        )

        # Speed up
        for key, fn in nonlinear_dict.items():
            nonlinear_dict[key] = jax.jit(fn)

        nonlinear_list = list(
            sort_dict_by_keys(nonlinear_dict, self.features_keys).values()
        )
        feats_list = chain_list_funcs(nonlinear_list, feats_list)
        return feats_list

    def _get_constraints_fn(self):
        constraints_dict = {}

        constraints_dict["offtrack"] = partial(constraints.is_offtrack, env=self)
        constraints_dict["overspeed"] = partial(
            constraints.is_overspeed, env=self, max_speed=1.0
        )
        constraints_dict["underspeed"] = partial(
            constraints.is_underspeed, env=self, min_speed=0.2
        )
        constraints_dict["uncomfortable"] = partial(
            constraints.is_uncomfortable, env=self, max_actions=self._control_bound
        )
        constraints_dict["wronglane"] = partial(
            constraints.is_wronglane, env=self, lane_idx=2
        )
        constraints_dict["collision"] = partial(constraints.is_collision, env=self)
        constraints_fn = merge_dict_funcs(constraints_dict)

        return constraints_dict, constraints_fn

    def _get_metadata_fn(self):
        metadata_dict = {}
        metadata_dict["overtake0"] = partial(
            constraints.is_overtake, env=self, car_idx=0
        )
        metadata_dict["overtake1"] = partial(
            constraints.is_overtake, env=self, car_idx=1
        )
        metadata_fn = merge_dict_funcs(metadata_dict)
        return metadata_dict, metadata_fn

    def update_key(self, rng_key):
        super().update_key(rng_key)
        self._task_sampler = seed(random_choice, rng_seed=rng_key)

    @property
    def features_keys(self):
        keys = super().features_keys
        keys += ["dist_objects"]
        return keys

    @property
    def constraints_keys(self):
        keys = super().constraints_keys
        # keys += ["hit_obstacle"]
        return keys


class Week6_01(HighwayDriveWorld_Week6):
    """Highway merging scenario, now with obstacles
    """

    def __init__(self):
        ## Boilerplate
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
        # Car states
        car1 = np.array([0.0, 0.3, np.pi / 2, 0])
        car2 = np.array([-lane_width, 0.9, np.pi / 2, 0])
        car_states = np.array([car1, car2])
        car_speeds = np.array([car_speed, car_speed])
        # Obstacle states
        # obstacle_states = np.array([[0.0, 0.3], [-lane_width, 0.3], [-lane_width, 0.5]])
        obstacle_states = np.array([[0.0, 0.3]])

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
            obstacle_states=obstacle_states,
        )
