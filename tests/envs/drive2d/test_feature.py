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


def test_diff_to_2d():
    x = np.repeat(np.array([[0, 0, np.pi / 2, 0]]), 5, axis=0)
    y = np.repeat(np.array([[1, 2, np.pi / 2, 0]]), 5, axis=0)
    result = np.repeat(np.array([[-1, -2]]), 5, axis=0)
    assert np.allclose(diff_to(x, y), result)


def test_diff_to_2d1d():
    x = np.repeat(np.array([[0, 0, np.pi / 2, 0]]), 5, axis=0)
    y = np.array([1, 2, np.pi / 2, 0])
    result = np.repeat(np.array([[-1, -2]]), 5, axis=0)
    assert np.allclose(diff_to(x, y), result)


def test_dist_to_segment():
    pt1 = np.array([1, 0])
    pt2 = np.array([0.5, 0])
    x = np.array([0.75, 1])
    result = np.array([1.0])
    assert np.allclose(dist_to_segment(x, pt1, pt2), result)


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

    data = np.array([-5, -3])
    mu = np.array([0.5, 0.1])
    sigma = np.array([0.1, 0.2])
    var = multivariate_normal(mean=mu, cov=np.diag(sigma ** 2))
    assert np.allclose(gaussian_feat(data, sigma, mu), [var.pdf(data)])
