from rdb.envs.drive2d.core.feature import *
import pytest
import jax.numpy as np


def test_dist_to():
    x = np.array([0, 0, np.pi / 2, 0])
    y = np.array([1, 2, np.pi / 2, 0])
    result = np.sqrt(np.array([5]))
    assert np.allclose(dist_to(x, y), result)


def test_diff_to():
    x = np.array([0, 0, np.pi / 2, 0])
    y = np.array([1, 2, np.pi / 2, 0])
    result = np.array([-1, -2])
    assert np.allclose(diff_to(x, y), result)


def test_diff_feat():
    x = np.array([1, 2])
    mu = np.array([3, 2])
    result = np.array([-2, 0])
    assert np.allclose(diff_feat(x, mu), result)
    x = np.array([1, 2])
    mu = 3
    result = np.array([-2, -1])
    assert np.allclose(diff_feat(x, mu), result)


def test_dist_to_lane():
    center = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    x = np.array([1.0, 0.0])
    result = np.array([1.0])
    assert np.allclose(dist_to_lane(center, normal, x), result)

    x = np.array([-1.0, 3.0])
    result = np.array([1.0])
    assert np.allclose(dist_to_lane(center, normal, x), result)


def test_dist_inside_fence():
    center = np.array([0.0, 0.0])
    normal = np.array([1.0, 0.0])
    x = np.array([1.0, 0.0])
    result = np.array([1.0])
    assert np.allclose(dist_inside_fence(center, normal, x), result)

    x = np.array([-1.0, 3.0])
    result = np.array([-1.0])
    assert np.allclose(dist_inside_fence(center, normal, x), result)


def test_quadratic():
    data = np.array([-1, -1])
    result = np.array([1, 1])
    assert np.allclose(quadratic_feat(data), result)

    data = np.array([-1, -1])
    goal = np.array([1, 1])
    result = np.array([4, 4])
    assert np.allclose(quadratic_feat(data, goal), result)


def test_gaussian():
    from scipy.stats import multivariate_normal

    data = np.array([-1, -1])
    mu = np.array([0.5, 0.5])
    sigma = np.array([1, 2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    assert np.allclose(gaussian_feat(data, sigma, mu), [var.pdf(data)])
    import pdb

    pdb.set_trace()

    data = np.array([-5, -3])
    mu = np.array([0.5, 0.1])
    sigma = np.array([0.1, 0.2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    assert np.allclose(gaussian_feat(data, sigma, mu), [var.pdf(data)])
