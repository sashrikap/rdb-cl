"""Constraints that indicate traffic violations.

Note:
    * Drive2d controller (rdb.optim.mpc.py) utilizes unconstrained optimization.
      These constraints are only used post-hoc to examine optimized results.

Use Cases:
    * For evaluating different acquisition function in active IRD.

TODO:
    * Jerk detection

"""

from rdb.envs.drive2d.core.feature import make_batch
import jax
import jax.numpy as np


def build_offtrack(env):
    """Detects when main_car nudges the edge of the track.

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)

    Return:
        * offtrack (ndarray): (T, nbatch) boolean

    Note:
        * Needs to know car shape a-priori

    """
    num_cars = len(env.cars)
    threshold = env.lane_width / 8
    vfn = jax.vmap(env.raw_features_dict["dist_fences"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats (T, nbatch, nfence, 1)
        feats = vfn(states, actions)
        assert len(feats.shape) == 4
        return np.any(np.abs(feats) > threshold, axis=(2, 3))

    return func


def build_collision(env):
    """Detects when main_car nudges another car.

    Args:
        actions (ndarray): (T, nbatch, udim)
        states (ndarray): (T, nbatch, xdim)

    Return:
        * collision (ndarray): (T, nbatch) boolean

    Note:
        * Needs to know car shape a-priori

    """
    ncars = len(env.cars)
    threshold = np.array([env.car_width, env.car_length])
    vfn = jax.vmap(env.raw_features_dict["dist_cars"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats: (T, nbatch, ncars, 2)
        feats = vfn(states, actions)
        assert len(feats.shape) == 4
        return np.any(np.all(np.abs(feats) < threshold, axis=3), axis=2)

    return func


def build_crash_objects(env):
    """Detects when main_car crashes into obstacle.

    Args:
        actions (ndarray): (T, nbatch, udim)
        states (ndarray): (T, nbatch, xdim)

    Return:
        * collision (ndarray): (T, nbatch) boolean

    Note:
        * Needs to know car shape a-priori

    """
    ncars = len(env.cars)
    threshold = np.array([env.car_width * 0.5, env.car_length])
    vfn = jax.vmap(env.raw_features_dict["dist_objects"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats: (T, nbatch, ncars, 2)
        feats = vfn(states, actions)
        assert len(feats.shape) == 4
        return np.any(np.all(np.abs(feats) < threshold, axis=3), axis=2)

    return func


def build_uncomfortable(env, max_actions):
    """Detects if actions exceed max actions.

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)

    Return:
        * uncomfortable (ndarray): (T, nbatch) boolean

    """

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        return np.any(np.abs(actions) > max_actions, axis=2)

    return func


def build_overspeed(env, max_speed):
    """Detects when main_car runs overspeed.

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)

    Return:
        * overspeed (ndarray): (T, nbatch) boolean

    """
    vfn = jax.vmap(env.raw_features_dict["speed"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats (T, nbatch, 1)
        feats = vfn(states, actions)
        assert len(feats.shape) == 3
        return np.any(feats > max_speed, axis=2)

    return func


def build_underspeed(env, min_speed):
    """Detects when main_car runs underspeed.

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)

    Return:
        * underspeed (ndarray): (T, nbatch) boolean

    """
    vfn = jax.vmap(env.raw_features_dict["speed"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats (T, nbatch, 1)
        feats = vfn(states, actions)
        assert len(feats.shape) == 3
        return np.any(feats < min_speed, axis=2)

    return func


def build_wronglane(env, lane_idx):
    """Detects when main_car (center) runs onto different lane.

    Note:
        * Needs to know lane shapes a-priori

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)
        lane_idx (tuple): wrong lane's index for `env.lanes[idx]`

    Return:
        * wronglane (ndarray): (T, nbatch) boolean

    """
    vfn = jax.vmap(env.raw_features_dict["dist_lanes"])

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats (T, nbatch, nlanes, 1)
        feats = vfn(states, actions)
        assert len(feats.shape) == 4
        return np.any(feats[:, :, lane_idx, None, :] < env.lane_width / 2, axis=(2, 3))

    return func


def build_overtake(env, car_idx):
    """Detects when main_car overtakes the car_idx-th car.

    Args:
        states (ndarray): (T, nbatch, xdim)
        actions (ndarray): (T, nbatch, udim)
        lane_idx (tuple): wrong lane's index for `env.lanes[idx]`

    Return:
        * overtake (ndarray): (T, nbatch) boolean

    """
    main_y_idx = 9
    car_y_idx = 4 * car_idx + 1

    def overtake_fn(state, actions):
        return state[:, main_y_idx, None] > state[:, car_y_idx, None]

    vfn = jax.vmap(overtake_fn)

    @jax.jit
    def func(states, actions):
        assert len(states.shape) == 3
        assert len(actions.shape) == 3
        # feats (T, nbatch, 1)
        feats = vfn(states, actions)
        assert len(feats.shape) == 3
        return np.any(feats, axis=2)

    return func
