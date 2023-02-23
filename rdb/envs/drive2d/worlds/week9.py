"""
One autonomous car is merging with obstacles.
"""


import jax
import jax.numpy as jnp
# import numpyro
# import numpyro.distributions as dist
import itertools, copy
from collections import OrderedDict
from rdb.optim.utils import *
from rdb.envs.drive2d.core import car, objects, constraints
from rdb.envs.drive2d.core.feature import *
from rdb.envs.drive2d.worlds.highway import HighwayDriveWorld
from functools import partial
# from numpyro.handlers import seed
from rdb.exps.utils import Profiler


class HighwayDriveWorld_Week9(HighwayDriveWorld):
    def __init__(
        self,
        main_state,
        goal_speed,
        goal_lane,
        # Control bound
        max_throttle=0.5,
        max_brake=0.4,
        max_steer=0.3,
        # States
        car_states=[],
        car_speeds=[],
        dt=0.1,
        horizon=10,
        num_lanes=3,
        lane_width=0.13,
        max_speed=0.9,
        car_ranges=[[-0.4, 1.0], [-0.4, 1.0]],
        car_delta=0.1,
        obstacle_states=[],
        tree_states=[],
        debris_states=[],
        obs_ranges=[[-0.16, 0.16, -0.8, 0.8]],
        obs_delta=[0.04, 0.1],
        task_naturalness="all",
    ):
        # Define cars
        cars = []
        for state, speed in zip(car_states, car_speeds):
            cars.append(car.FixSpeedCar(self, jnp.array(state), speed))
        main_car = car.OptimalControlCar(self, main_state, horizon)
        self._goal_speed = goal_speed
        self._goal_lane = goal_lane
        self._max_speed = max_speed

        # Control bound
        self._max_throttle = max_throttle
        self._max_steer = max_steer
        self._max_brake = max_brake
        main_car.max_throttle = max_throttle
        main_car.max_steer = max_steer
        main_car.max_brake = max_brake
        main_car.max_speed = max_speed

        # Define objects
        objs = []
        for state in obstacle_states:
            objs.append(objects.Obstacle(jnp.array(state)))

        # Add trees
        for state in tree_states:
            objs.append(objects.Tree(jnp.array(state)))
        
        # Add debris
        for state in debris_states:
            objs.append(objects.Debris(jnp.array(state)))

        super().__init__(main_car, cars, num_lanes=num_lanes, objects=objs, dt=dt)

        # Define all tasks to sample from
        self._task_sampler = None
        self._task_naturalness = task_naturalness
        self._car_ranges = car_ranges
        self._car_delta = car_delta
        self._obs_ranges = obs_ranges
        self._obs_delta = obs_delta

    def set_task(self, task):
        state = self.get_init_state(task)
        self.set_init_state(state)

    def get_init_state(self, task):
        """Get task initial state without modifying environment internal variables.

        Effect:
            * env.reset

        """
        return self.get_init_states([task])

    def get_init_states(self, tasks):
        """Vectorized version of `get_init_state`
        """
        tasks = onp.array(tasks)
        assert tasks.shape[-1] == 2 + 2 * len(self._objects), "Task format incorrect"
        obj_idx = 8
        state = copy.deepcopy(self.state)
        for ci, car in enumerate(self.cars + [self._main_car]):
            state = state.at[:, ci * 4 : (ci + 1) * 4].set(car.init_state)
        all_states = onp.tile(onp.array(state), (len(tasks), 1))
        # Object state
        state_idx, task_idx = 0, 0 #0, 2
        for obj in self._objects:
            next_task_idx = task_idx + obj.xdim
            next_state_idx = state_idx + obj.xdim
            all_states[:, state_idx:next_state_idx] = onp.array(
                tasks[:, task_idx:next_task_idx]
            )
            task_idx = next_task_idx
            state_idx = next_state_idx
        return all_states

    def _get_nonlinear_features_dict(self, feats_dict):
        """
        Args:
            feats_dict (dict): dict of environment feature functions

        Note:
            * Gaussian : exp(-dist^2/sigma^2)
            * Exponent : exp(run_over)

        """
        max_feats_dict = OrderedDict()
        nlr_feats_dict = OrderedDict()
        # Sum across ncars & state (nbatch, ncars, state)
        sum_items = partial(jnp.sum, axis=(1, 2))
        # Sum across state (nbatch, state)
        sum_state = partial(jnp.sum, axis=1)
        fence_const = 3.0
        ctrl_const = 3.0
        speed_const = 5.0

        ## Car distance feature
        ncars, car_dim = len(self._cars), 2
        sigcar = jnp.array([self._car_width / 2, self._car_length])
        # nlr_feats_dict["dist_cars"] = compose(
        #     sum_items, partial(gaussian_feat, sigma=sigcar)
        # )
        # # max_feats_dict["dist_cars"] = jnp.sum(gaussian_feat(jnp.zeros((car_dim, ncars, car_dim)), sigma=sigcar))
        # max_feats_dict["dist_cars"] = jnp.sum(
        #     gaussian_feat(jnp.zeros((car_dim, 1, car_dim)), sigma=sigcar)
        # )

        ## Lane distance feature
        nlr_feats_dict["dist_lanes"] = compose(
            sum_items,
            neg_feat,
            partial(gaussian_feat, sigma=self._car_length),
            partial(item_index_feat, index=self._goal_lane),
        )
        max_feats_dict["dist_lanes"] = 0.0

        ## Fence distance feature
        max_fence = fence_const * self._lane_width
        nlr_feats_dict["dist_fences"] = compose(
            sum_items,
            # quadratic_feat, # neg_relu_feat,
            partial(relu_feat, max_val=max_fence),
            lambda dist: dist,
        )
        max_feats_dict["dist_fences"] = max_fence

        ## Object distance feature
        nobjs, obj_dim = len(self._objects), 2
        sigobj = jnp.array([self._car_width / 2, self._car_length * 2])
        
        if self._objects:
            nlr_feats_dict["dist_obstacles"] = compose(
                sum_items, partial(gaussian_feat, sigma=sigobj)
            )
            nlr_feats_dict["dist_trees"] = compose(
                sum_items, partial(gaussian_feat, sigma=sigobj)
            )
            nlr_feats_dict["dist_debris"] = compose(
                sum_items, partial(gaussian_feat, sigma=sigobj)
            )
            max_feats_dict["dist_obstacles"] = jnp.sum(
                gaussian_feat(jnp.zeros((obj_dim, 1, obj_dim)), sigma=sigobj)
            )
            max_feats_dict["dist_trees"] = jnp.sum(
                gaussian_feat(jnp.zeros((obj_dim, 1, obj_dim)), sigma=sigobj)
            )
            max_feats_dict["dist_debris"] = jnp.sum(
                gaussian_feat(jnp.zeros((obj_dim, 1, obj_dim)), sigma=sigobj)
            )

        ## Control features
        ones = jnp.ones((1, 1))  # (nbatch=1, dim=1)
        max_control = jnp.array([self._max_steer, self._max_throttle]) * ctrl_const
        nlr_feats_dict["control"] = compose(
            sum_state, partial(quadratic_feat, max_val=max_control)
        )
        max_feats_dict["control"] = jnp.sum(quadratic_feat(ones * max_control))
        nlr_feats_dict["control_throttle"] = compose(
            sum_state, partial(quadratic_feat, max_val=self._max_throttle * ctrl_const)
        )
        max_feats_dict["control_throttle"] = jnp.sum(
            quadratic_feat(ones * self._max_throttle * ctrl_const)
        )
        nlr_feats_dict["control_brake"] = compose(
            sum_state, partial(quadratic_feat, max_val=self._max_brake * ctrl_const)
        )
        max_feats_dict["control_brake"] = jnp.sum(
            quadratic_feat(ones * self._max_brake * ctrl_const)
        )
        nlr_feats_dict["control_turn"] = compose(
            sum_state, partial(quadratic_feat, max_val=self._max_steer * ctrl_const)
        )
        max_feats_dict["control_turn"] = jnp.sum(
            quadratic_feat(ones * self._max_steer * ctrl_const)
        )

        ## Speed features
        max_dspeed = speed_const * (self._max_speed - self._goal_speed)
        nlr_feats_dict["speed"] = compose(
            sum_state,
            partial(quadratic_feat, goal=self._goal_speed, max_val=max_dspeed),
        )
        max_feats_dict["speed"] = jnp.sum(quadratic_feat(ones * max_dspeed))
        nlr_feats_dict["speed_over"] = compose(
            sum_state,
            partial(quadratic_feat, max_val=max_dspeed),
            partial(more_than, y=self._goal_speed),
        )
        max_feats_dict["speed_over"] = jnp.sum(quadratic_feat(ones * max_dspeed))
        nlr_feats_dict["speed_under"] = compose(
            sum_state,
            partial(quadratic_feat, max_val=max_dspeed),
            partial(less_than, y=self._goal_speed),
        )
        max_feats_dict["speed_under"] = jnp.sum(quadratic_feat(ones * max_dspeed))

        nlr_feats_dict["bias"] = identity_feat
        max_feats_dict["bias"] = ones

        nlr_feats_dict = chain_dict_funcs(nlr_feats_dict, feats_dict)

        ## JAX compile
        for key, fn in nlr_feats_dict.items():
            nlr_feats_dict[key] = jax.jit(fn)

        return nlr_feats_dict, max_feats_dict

    def _build_constraints_fn(self):
        constraints_dict = OrderedDict()
        constraints_dict["offtrack"] = constraints.build_offtrack(env=self)
        constraints_dict["overspeed"] = constraints.build_overspeed(
            env=self, max_speed=self._max_speed
        )
        constraints_dict["underspeed"] = constraints.build_underspeed(
            env=self, min_speed=-0.1
        )
        constraints_dict["uncomfortable"] = constraints.build_uncomfortable(
            env=self,
            max_throttle=self._max_throttle,
            max_brake=self._max_brake,
            max_steer=self._max_steer,
        )
        constraints_dict["wronglane"] = constraints.build_wronglane(
            env=self, lane_idx=2
        )
        # Commenting out since there are no other cars
        # constraints_dict["collision"] = constraints.build_collision(env=self)
        constraints_dict["crash_objects"] = constraints.build_crash_objects(env=self)
        self._constraints_fn = merge_dict_funcs(constraints_dict)
        self._constraints_dict = constraints_dict

    def _build_metadata_fn(self):
        metadata_dict = OrderedDict()
        metadata_dict["overtake0"] = constraints.build_overtake(env=self, car_idx=0)
        metadata_dict["overtake1"] = constraints.build_overtake(env=self, car_idx=1)
        self._metadata_fn = merge_dict_funcs(metadata_dict)
        self._metadata_dict = metadata_dict

    def update_key(self, rng_key):
        super().update_key(rng_key)

    def _setup_tasks(self):
        obs_ranges = self._obs_ranges
        obs_delta = self._obs_delta
        self._grid_tasks = []
        for ci, car_range in enumerate(self._car_ranges):
            self._grid_tasks.append(
                jnp.arange(car_range[0], car_range[1], self._car_delta)
            )
        for oi in range(len(self._objects)):
            obs_range_x = jnp.arange(obs_ranges[oi][0], obs_ranges[oi][1], obs_delta[0])
            obs_range_y = jnp.arange(obs_ranges[oi][2], obs_ranges[oi][3], obs_delta[1])
            self._grid_tasks.append(obs_range_x)
            self._grid_tasks.append(obs_range_y)
        all_tasks = list(itertools.product(*self._grid_tasks))
        self._all_tasks = self._get_natural_tasks(all_tasks)
        self._all_task_difficulties = self._get_task_difficulties(self._all_tasks)

    def _get_natural_tasks(self, tasks):
        """Filter out tasks that are not natural (keep tasks where initial positions
        of other cars and objects are far).

        """
        if self._task_naturalness == "all":
            all_tasks = tasks
        elif self._task_naturalness == "distance":
            ## Difference to cars and objects
            all_states = self.get_init_states(tasks)
            all_acs = jnp.zeros((len(tasks), 2))

            diff_cars = self._raw_features_dict["dist_cars"](all_states, all_acs)
            diff_objs = self._raw_features_dict["dist_objects"](all_states, all_acs)

            # diff_cars = diff_cars.reshape(-1, len(self._cars), 2)
            diff_objs = diff_objs.reshape(-1, len(self._objects), 2)

            head_length = 2 * self._car_length
            back_length = self._car_length
            car_width = self._car_width

            # Whether any car is too close
            cars_x_too_close = onp.logical_and(
                diff_cars[:, :, 0] < car_width, diff_cars[:, :, 0] > -car_width
            )
            cars_y_too_close = onp.logical_and(
                diff_cars[:, :, 1] < head_length, diff_cars[:, :, 1] > -back_length
            )
            cars_too_close = onp.any(
                onp.logical_and(cars_x_too_close, cars_y_too_close), axis=-1
            )
            # Whether any object is too close
            objs_x_too_close = onp.logical_and(
                diff_objs[:, :, 0] < car_width, diff_objs[:, :, 0] > -car_width
            )
            objs_y_too_close = onp.logical_and(
                diff_objs[:, :, 1] < head_length, diff_objs[:, :, 1] > -back_length
            )
            objs_too_close = onp.any(
                onp.logical_or(objs_x_too_close, objs_y_too_close), axis=-1
            )

            too_close = onp.logical_or(cars_too_close, objs_too_close)
            all_tasks = onp.array(tasks)[onp.logical_not(onp.array(too_close))]
            # all_tasks = onp.array(tasks)[onp.array(too_close)]
        else:
            raise NotImplementedError
        return all_tasks

    def _get_task_difficulties(self, tasks, method="mean_inv"):
        """ Compute task difficulties.
        """
        if method == "mean_inv":
            ## Mean of inverse of object distances
            ## Difference to cars and objects
            all_states = self.get_init_states(tasks)
            all_acs = jnp.zeros((len(tasks), 2))

            diff_cars = self._raw_features_dict["dist_cars"](all_states, all_acs)
            diff_objs = self._raw_features_dict["dist_objects"](all_states, all_acs)

            diff_cars = diff_cars.reshape(-1, len(self._cars), 2)
            diff_objs = diff_objs.reshape(-1, len(self._objects), 2)

            dist_cars = jnp.linalg.norm(diff_cars, axis=2)
            dist_objs = jnp.linalg.norm(diff_objs, axis=2)

            sum_invs = jnp.sum((1 / dist_cars), axis=1) + jnp.sum(
                (1 / dist_objs), axis=1
            )
            mean_invs = sum_invs / float(len(self._cars) + len(self._objects))
            all_tasks = onp.array(tasks)[onp.array(too_close)]
            difficulties = mean_invs
        else:
            raise NotImplementedError
        return difficulties


class Week9_01(HighwayDriveWorld_Week9):
    """
    Highway merging scenario with one obstacle, a cone to the right and
    ahead of the car.
    """

    def __init__(self):
        ## Boilerplate
        main_speed = 0.7
        car_speed = 0.5
        main_state = jnp.array([0, 0, jnp.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        # Lane size
        lane_width = 0.13
        num_lanes = 3
        # Obstacle states
        obstacle_states = jnp.array([[lane_width, 1]])
        # Don't filter any task
        task_naturalness = "all"

        super().__init__(
            main_state,
            goal_speed=goal_speed,
            goal_lane=goal_lane,
            dt=dt,
            horizon=horizon,
            num_lanes=num_lanes,
            lane_width=lane_width,
            obstacle_states=obstacle_states,
            task_naturalness=task_naturalness,
        )

class Week9_02(HighwayDriveWorld_Week9):
    """
    Highway merging scenario with one obstacle, a tree to the right and
    ahead of the car.
    """

    def __init__(self):
        ## Boilerplate
        main_speed = 0.7
        car_speed = 0.5
        main_state = jnp.array([0, 0, jnp.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        # Lane size
        lane_width = 0.13
        num_lanes = 3
        # Tree states
        tree_states = jnp.array([[0.5, 1]])
        # Don't filter any task
        task_naturalness = "all"

        super().__init__(
            main_state,
            goal_speed=goal_speed,
            goal_lane=goal_lane,
            dt=dt,
            horizon=horizon,
            num_lanes=num_lanes,
            lane_width=lane_width,
            # obstacle_states=obstacle_states,
            tree_states=tree_states,
            task_naturalness=task_naturalness,
        )

class Week9_03(HighwayDriveWorld_Week9):
    """
    Highway merging scenario with one obstacle, a fallen tree to the right and
    ahead of the car.
    """

    def __init__(self):
        ## Boilerplate
        main_speed = 0.7
        car_speed = 0.5
        main_state = jnp.array([0, 0, jnp.pi / 2, main_speed])
        goal_speed = 0.8
        goal_lane = 0
        horizon = 10
        dt = 0.25
        # Lane size
        lane_width = 0.13
        num_lanes = 3
        # Debris states
        debris_states = jnp.array([[0, 1]])
        # Don't filter any task
        task_naturalness = "all"

        super().__init__(
            main_state,
            goal_speed=goal_speed,
            goal_lane=goal_lane,
            dt=dt,
            horizon=horizon,
            num_lanes=num_lanes,
            lane_width=lane_width,
            debris_states=debris_states,
            task_naturalness=task_naturalness,
        )       