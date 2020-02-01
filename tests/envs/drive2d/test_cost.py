"""Test combined dynamics features and costs.
"""
from rdb.envs.drive2d.core.dynamics import *
from rdb.envs.drive2d.core import feature
from rdb.optim.utils import *
from rdb.infer import *
import itertools
import numpy as onp
import pytest


features_keys = ["dist_cars", "dist_lanes", "speed", "control", "control_thrust"]


def build_dynamics(n_cars):
    car_dyns, idxs = [], []
    for i in range(n_cars):
        car_dyns.append(build_car_dynamics(0.2))
        idxs.append((4 * i, 4 * i + 4))

    lane_ctr = np.array([1, 0])
    lane_normal = np.array([2, 2])
    lane_dyn = lambda state, actions: feature.dist_to_lane(
        state[..., 0:4], lane_ctr, lane_normal
    )

    normals = np.array([[1.0, 0.0], [-1.0, 0.0]])
    fence_normal = np.array([1, 0])
    fence_ctr = np.array([2, 0])
    fence_dyn = lambda state, actions: feature.dist_outside_fence(
        state[..., 0:4], fence_ctr, fence_normal
    )

    fns = [index_func(dyn, idx) for dyn, idx in zip(car_dyns, idxs)]
    dyn_fn = concat_funcs(fns + [lane_dyn, lane_dyn, fence_dyn], axis=1)
    return dyn_fn


def build_features(dyn_fn, n_cars):
    main_idx = (0, 4)
    car_idxs = []
    lane_ctr = np.array([1, 0])
    lane_normal = np.array([2, 2])

    car_feat = lambda state, actions: feature.diff_to(state[..., 4:8], state[..., 0:4])
    lane_feat = lambda state, actions: feature.dist_to_lane(
        state[..., 0:4], lane_ctr, lane_normal
    )
    car_fn = concat_funcs([car_feat] * n_cars, axis=1)
    lane_fn = concat_funcs([lane_feat, lane_feat], axis=1)

    speed_fn = lambda state, actions: feature.speed_size(state[..., 0:4])
    control_mag = lambda state, actions: feature.control_magnitude(actions)
    control_thrust = lambda state, actions: feature.control_thrust(actions)
    merged_dict = dict(
        dist_cars=car_fn,
        dist_lanes=lane_fn,
        speed=speed_fn,
        control=control_mag,
        control_thrust=control_thrust,
    )
    merged_fn = merge_dict_funcs(merged_dict)
    return merged_dict, merged_fn


def build_nonlinear(feats_dict, n_cars):
    nlr_feats_dict = {}
    nlr_feats_dict["dist_cars"] = compose(
        partial(np.sum, axis=1),
        partial(feature.gaussian_feat, sigma=np.array([0.4, 0.2]).repeat(n_cars)),
    )
    nlr_feats_dict["dist_lanes"] = compose(
        partial(np.sum, axis=1),
        feature.neg_feat,
        partial(feature.gaussian_feat, sigma=1),
        partial(feature.index_feat, index=0),
    )
    nlr_feats_dict["speed"] = compose(
        partial(np.sum, axis=1), partial(feature.quadratic_feat, goal=2)
    )
    nlr_feats_dict["control"] = compose(partial(np.sum, axis=1), feature.quadratic_feat)
    nlr_feats_dict["control_thrust"] = compose(
        partial(np.sum, axis=1), feature.quadratic_feat
    )
    nlr_feats_dict = sort_dict_by_keys(nlr_feats_dict, feats_dict.keys())
    nlr_feats_dict = chain_dict_funcs(nlr_feats_dict, feats_dict)
    merged_fn = merge_dict_funcs(nlr_feats_dict)
    return nlr_feats_dict, merged_fn


def build_cost(nlr_dict):
    cost_runtime = weigh_funcs_runtime(nlr_dict)
    return cost_runtime


@pytest.mark.parametrize("batch,n_cars", list(itertools.product([1, 2, 10], [2, 4, 5])))
def test_combined_full(batch, n_cars):
    dyn_fn = build_dynamics(n_cars)
    xcar = np.array([0, 0, np.pi / 2, 0]).tile((batch, 1))
    ucar = np.array([0, 1]).tile((batch, 1))
    xn = xcar.repeat(n_cars, axis=1)
    out = dyn_fn(xn, ucar)
    assert out.shape == (batch, 4 * n_cars + 3)


@pytest.mark.parametrize("batch,n_cars", list(itertools.product([1, 2, 10], [2, 4, 5])))
def test_raw_features_full(batch, n_cars):
    dyn_fn = build_dynamics(n_cars)
    feats_dict, feats_fn = build_features(dyn_fn, n_cars)
    xcar = np.array([0, 0, np.pi / 2, 0]).tile((batch, 1))
    ucar = np.array([0, 1]).tile((batch, 1))
    xn = xcar.repeat(n_cars, axis=1)
    out = feats_fn(xn, ucar)
    for key, val in out.items():
        assert val.shape[0] == batch
        assert len(val.shape) == 2


@pytest.mark.parametrize("batch,n_cars", list(itertools.product([1, 2, 10], [2, 4, 5])))
def test_nonlinear_features_full(batch, n_cars):
    dyn_fn = build_dynamics(n_cars)
    feats_dict, feats_fn = build_features(dyn_fn, n_cars)
    nlr_dict, nlr_fn = build_nonlinear(feats_dict, n_cars)
    xcar = np.array([0, 0, np.pi / 2, 0]).tile((batch, 1))
    ucar = np.array([0, 1]).tile((batch, 1))
    xn = xcar.repeat(n_cars, axis=1)
    out = nlr_fn(xn, ucar)
    for key, val in out.items():
        assert val.shape[0] == batch
        assert len(val.shape) == 1


@pytest.mark.parametrize("batch,n_cars", list(itertools.product([1, 2, 10], [2, 4, 5])))
def test_nonlinear_featrues_cost(batch, n_cars):
    dyn_fn = build_dynamics(n_cars)
    feats_dict, feats_fn = build_features(dyn_fn, n_cars)
    nlr_dict, nlr_fn = build_nonlinear(feats_dict, n_cars)
    xcar = np.array([0, 0, np.pi / 2, 0]).tile((batch, 1))
    ucar = np.array([0, 1]).tile((batch, 1))
    xn = xcar.repeat(n_cars, axis=1)
    out_dict = nlr_fn(xn, ucar)
    weights = {}
    for key in out_dict.keys():
        weights[key] = onp.random.random(batch)
    weights_arr = DictList(weights).prepare(features_keys).numpy_array()
    cost_fn = build_cost(nlr_dict)
    out = cost_fn(xn, ucar, weights=weights_arr)
    assert out.shape[0] == batch
    assert len(out.shape) == 1
