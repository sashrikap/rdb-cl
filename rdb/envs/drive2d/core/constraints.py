"""Constraints that indicate traffic violations.

Note:
    * Drive2d controller (rdb.optim.mpc.py) utilizes unconstrained optimization.
      These constraints are only used post-hoc to examine optimized results.

Use Cases:
    * For evaluating different acquisition function in active IRD.

TODO:
    * Currently not using JIT to speed up
    * Collision assumes cars are the same

"""

from rdb.envs.drive2d.core.feature import make_batch
import jax
import jax.numpy as np


def is_offtrack(states, actions, env):
    """Detects when main_car nudges the edge of the track.

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)

    Return:
        * offtrack (ndarray): (T, ) boolean

    Note:
        * Needs to know car shape a-priori

    """
    num_cars = len(env.cars)
    threshold = 0.5 * env.car_width + 0.5 * env.lane_width
    fn = env.raw_features_dict["dist_fences"]
    feats = []
    states, actions = make_batch(states), make_batch(actions)
    for s, a in zip(states, actions):
        feats.append(fn(s, a))
    feats = np.array(feats)
    return np.any(np.abs(feats) < threshold, axis=1)


def is_collision(states, actions, env):
    """Detects when main_car nudges another car.

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)

    Return:
        * collision (ndarray): (T, ) boolean

    Note:
        * Needs to know car shape a-priori

    """
    num_cars = len(env.cars)
    threshold = np.array([env.car_width, env.car_length])
    fn = env.raw_features_dict["dist_cars"]
    feats = []
    states, actions = make_batch(states), make_batch(actions)
    for s, a in zip(states, actions):
        feats.append(fn(s, a))
    feats = np.array(feats)
    per_car = np.all(np.abs(feats) < threshold, axis=-1)
    return np.any(per_car, axis=1)


def is_uncomfortable(states, actions, env, max_actions):
    """Detects if actions exceed max actions.

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)

    Return:
        * uncomfortable (ndarray): (T, ) boolean

    """
    actions = make_batch(actions)
    return np.any(np.abs(actions) > max_actions, axis=1)


# @jax.jit
def is_overspeed(states, actions, env, max_speed):
    """Detects when main_car runs overspeed.

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)

    Return:
        * overspeed (ndarray): (T, ) boolean

    """
    fn = env.raw_features_dict["speed"]
    feats = []
    states, actions = make_batch(states), make_batch(actions)
    for s, a in zip(states, actions):
        feats.append(fn(s, a))
    feats = np.array(feats)
    return np.any(feats > max_speed, axis=1)


# @jax.jit
def is_underspeed(states, actions, env, min_speed):
    """Detects when main_car runs underspeed.

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)

    Return:
        * underspeed (ndarray): (T, ) boolean

    """
    fn = env.raw_features_dict["speed"]
    feats = []
    states, actions = make_batch(states), make_batch(actions)
    for s, a in zip(states, actions):
        feats.append(fn(s, a))
    feats = np.array(feats)
    return np.any(feats < min_speed, axis=1)


# @jax.jit
def is_wronglane(states, actions, env, lane_idx):
    """Detects when main_car (center) runs onto different lane.

    Note:
        * Needs to know lane shapes a-priori

    Args:
        states (ndarray): (T, xdim)
        actions (ndarray): (T, udim)
        lane_idx (tuple): wrong lane's index for `env.lanes[idx]`

    Return:
        * wronglane (ndarray): (T, ) boolean

    """
    fn = env.raw_features_dict["dist_lanes"]
    feats = []
    states, actions = make_batch(states), make_batch(actions)
    for s, a in zip(states, actions):
        feats.append(fn(s, a))
    feats = np.array(feats)
    return feats[:, lane_idx] < env.lane_width / 2


def is_overtake(states, actions, env, car_idx):
    """Detects when main_car overtakes the car_idx-th car.
    """
    main_y_idx = 9
    car_y_idx = 4 * car_idx + 1

    def overtake_fn(state, actions):
        return state[main_y_idx] > state[car_y_idx]

    feats = []
    for s, a in zip(states, actions):
        feats.append(overtake_fn(s, a))
    feats = np.array(feats)
    return feats
