from rdb.envs.drive2d.core.feature import *
import jax.numpy as np
import pytest

batch = 10
x0 = np.array([0, 0, np.pi / 2, 0]).tile((batch, 1))
y0 = np.array([1, 2, np.pi / 2, 0]).tile((batch, 1))
pt = np.array([1, 2])
obj = np.array([1, 2, 3, 4])
idxs = np.zeros(batch)
u0 = np.array([1, 2]).tile((batch, 1))
xdim = 4


def run_one_act_feature(feat, u0):
    """Iterate on only u0"""
    ft_batch = feat(u0)
    assert len(ft_batch.shape) == 2
    ft_single = []
    for i, u in enumerate(u0):
        out = feat(np.array([u]))
        assert len(out.shape) == 2
        assert np.allclose(ft_batch[i], out)


def run_one_state_feature(feat, x0, u0):
    """Iterate on only x0 and not u0"""
    ft_batch = feat(x0, u0)
    assert len(ft_batch.shape) == 2
    ft_single = []
    for i, x in enumerate(x0):
        out = feat(np.array([x]), u0)
        assert len(out.shape) == 2
        assert np.allclose(ft_batch[i], out)


def run_two_state_feature(feat, x0, u0):
    """Iterate on both x0 and u0"""
    ft_batch = feat(x0, u0)
    assert len(ft_batch.shape) == 2
    ft_single = []
    for i, (x, u) in enumerate(zip(x0, u0)):
        out = feat(np.array([x]), np.array([u]))
        assert len(out.shape) == 2
        assert np.allclose(ft_batch[i], out)


def run_three_state_feature(feat, a0, b0, c0):
    """Only iterate on a0, not on b0, c0"""
    ft_batch = feat(x0, b0, c0)
    assert len(ft_batch.shape) == 2
    ft_single = []
    for i, a in enumerate(a0):
        out = feat(np.array([a]), b0, c0)
        assert len(out.shape) == 2
        assert np.allclose(ft_batch[i], out)


def run_four_state_feature(feat, a0, b0, c0, d0):
    """Only iterate on a0, not on b0, c0, d0"""
    ft_batch = feat(x0, b0, c0, d0)
    assert len(ft_batch.shape) == 2
    ft_single = []
    for i, a in enumerate(a0):
        out = feat(np.array([a]), b0, c0, d0)
        assert len(out.shape) == 2
        assert np.allclose(ft_batch[i], out)


# ====================================================
# ============ Environmental Features ================
# ====================================================


def test_dist_to():
    run_two_state_feature(dist_to, x0, y0)


def test_dist_to_segment():
    run_three_state_feature(dist_to_segment, x0, pt, pt)


def test_diff_to():
    run_one_state_feature(diff_to, x0, obj)
    run_two_state_feature(diff_to, x0, y0)


def test_dist_to_lane():
    center = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    x = np.array([[1.0, 0.0, 0, 0]])
    result = np.array([[1.0]])
    assert np.allclose(dist_to_lane(x, center, normal), result)

    x = np.array([[-1.0, 3.0, 0, 0]])
    result = np.array([[1.0]])
    assert np.allclose(dist_to_lane(x, center, normal), result)

    run_three_state_feature(dist_to_lane, x0, center, normal)


def test_dist_inside_fence():
    run_three_state_feature(dist_inside_fence, x0, pt, pt)
    center = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    x = np.array([[1.0, 0.0, 0, 0]])
    result = np.array([[1.0]])
    assert np.allclose(dist_inside_fence(x, center, normal), result)

    x = np.array([[-1.0, 3.0, 0, 0]])
    result = np.array([[-1.0]])
    assert np.allclose(dist_inside_fence(x, center, normal), result)


def test_dist_outside_fence():
    run_three_state_feature(dist_outside_fence, x0, pt, pt)


def test_speed_forward():
    run_one_act_feature(speed_forward, u0)


def test_speed_size():
    run_one_act_feature(speed_size, x0)


def test_control_magnitude():
    run_one_act_feature(control_magnitude, u0)


def test_control_thrust():
    run_one_act_feature(control_thrust, u0)


def test_control_brake():
    run_one_act_feature(control_brake, u0)


def test_control_turn():
    run_one_act_feature(control_turn, u0)


# ====================================================
# ============== Numerical Features ==================
# ====================================================


def test_index_feat():
    data = np.array([[1, 2], [3, 4]])
    index = 1
    result = np.array([[2], [4]])
    assert np.allclose(index_feat(data, index), result)
    index = 0
    result = np.array([[1], [3]])
    assert np.allclose(index_feat(data, index), result)
    run_one_state_feature(index_feat, x0, 0)


def test_quadratic():
    data = np.array([[-1, -1]])
    result = np.array([[1, 1]])
    assert np.allclose(quadratic_feat(data), result)

    data = np.array([[-1, -1]])
    goal = np.array([[1, 1]])
    result = np.array([[4, 4]])
    assert np.allclose(quadratic_feat(data, goal), result)
    run_one_act_feature(quadratic_feat, x0)


def test_abs_feat():
    x = np.array([[1, 2]])
    mu = np.array([3, 2])
    result = np.array([[2, 0]])
    assert np.allclose(abs_feat(x, mu), result)
    x = np.array([[1, 2]])
    mu = 3
    result = np.array([[2, 1]])
    assert np.allclose(abs_feat(x, mu), result)
    run_one_state_feature(abs_feat, x0, obj)


def test_neg_feat():
    x = np.array([[1, 2]])
    result = np.array([[-1, -2]])
    assert np.allclose(neg_feat(x), result)
    run_one_act_feature(neg_feat, x0)


def test_positive_const_feat():
    x = np.array([[2, -2]])
    result = np.array([[1, 0]])
    assert np.allclose(positive_const_feat(x), result)
    run_one_act_feature(positive_const_feat, x0)


def test_relu_feat():
    x = np.array([[2, -2]])
    result = np.array([[2, 0]])
    assert np.allclose(relu_feat(x), result)
    run_one_act_feature(relu_feat, x0)


def test_neg_relu_feat():
    x = np.array([[2, -2]])
    result = np.array([[0, -2]])
    assert np.allclose(neg_relu_feat(x), result)
    run_one_act_feature(neg_relu_feat, x0)


def test_sigmoid_feat():
    run_one_act_feature(sigmoid_feat, x0)
    run_one_state_feature(sigmoid_feat, x0, obj)


def test_neg_exp_feat():
    run_one_state_feature(neg_exp_feat, x0, obj)


def test_gaussian_feat():
    from scipy.stats import multivariate_normal

    data = np.array([[-1, -1]])
    mu = np.array([0.5, 0.5])
    sigma = np.array([1, 2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    result = np.array([var.pdf(data)])[:, None]
    assert np.allclose(gaussian_feat(data, sigma, mu), result)

    data = np.array([[-5, -3]])
    mu = np.array([0.5, 0.1])
    sigma = np.array([0.1, 0.2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    result = np.array([var.pdf(data)])[:, None]
    assert np.allclose(gaussian_feat(data, sigma, mu), result)

    data = np.array([[-1, -1], [-4, 2]])
    mu = np.array([0.5, 0.5])
    sigma = np.array([1, 2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    result = var.pdf(data)[:, None]
    assert np.allclose(gaussian_feat(data, sigma, mu), result)

    run_three_state_feature(gaussian_feat, x0, obj, obj)


def test_exp_bounded_feat():
    run_four_state_feature(exp_bounded_feat, x0, obj, obj, 4)
