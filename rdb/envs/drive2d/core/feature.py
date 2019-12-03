import jax.numpy as np
import jax

"""
TODO:
[0] Tweak gaussian features
[1] quadratic_feat broadcast not good for batch mode
[2] bounded_feat not good for batch mode
"""


@jax.jit
def make_batch(state):
    if type(state) == list:
        state = np.asarray(state)
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)
    return np.asarray(state)


# Environment specific
@jax.jit
def dist_to(x, target):
    """ sqrt((target1 - x1)^2 + (target2 - x2)^2)"""
    x = make_batch(x)
    target = make_batch(target)
    diff = np.array(x[..., :2]) - np.array(target[..., :2])
    return np.linalg.norm(diff, axis=-1)


@jax.jit
def dist_to_segment(x, pt1, pt2):
    """ Distance to line segment pt1-pt2 """
    x = make_batch(x)[..., :2]
    pt1 = make_batch(pt1)[..., :2]
    pt2 = make_batch(pt2)[..., :2]
    delta = pt2 - pt1
    sum_sqr = np.sum(np.square(delta), axis=-1)
    u = np.sum(delta * (x - pt1) / sum_sqr, axis=-1)
    u = np.clip(u, 0, 1.0)
    dx = pt1 + u * delta - x
    return np.linalg.norm(dx, axis=-1)


@jax.jit
def divide_by(x, y):
    """ (x1/y1, x2/y2) """
    x = make_batch(x)
    y = make_batch(y)
    return x / y


@jax.jit
def sum_square(x):
    """ x1 * x1 + x2 * x2 """
    x = make_batch(x)
    return np.sum(x * x, keepdims=True)


@jax.jit
def diff_to(x, y):
    """ (y1 - x1, y2 - x2)"""
    x = make_batch(x)
    y = make_batch(y)
    diff = np.array(x[..., :2]) - np.array(y[..., :2])
    return diff


@jax.jit
def dist_to_lane(center, normal, x):
    """ lane.normal @ diff_to_lane """
    x = make_batch(x)
    diff = np.array(center) - np.array(x[..., :2])
    return np.abs(np.sum(normal * diff, axis=-1))


@jax.jit
def dist_inside_fence(center, normal, x):
    x = make_batch(x)
    diff = np.array(x[..., :2]) - np.array(center)
    return np.sum(normal * diff, axis=-1)


@jax.jit
def diff_to_fence(center, normal, x):
    """
    Params
    : center : (2,)
    : normal : (2,)
    : x      : (nbatch, 2)
    Outout
    : diff   : (nbatch, 2)
    """
    x = make_batch(x)
    diff = np.array(x[..., :2]) - np.array(center)
    diff = normal * diff
    angle = np.array(x[..., 2])
    vh = np.concatenate([np.sin(angle), -np.cos(angle)], axis=-1)
    vv = np.concatenate([np.cos(angle), np.sin(angle)], axis=-1)
    diff_h = np.sum(diff * vh, axis=-1, keepdims=True)
    diff_v = np.sum(diff * vv, axis=-1, keepdims=True)
    return np.concatenate([diff_h, diff_v], axis=-1)


@jax.jit
def speed_forward(state):
    state = make_batch(state)
    return state[..., 3] * np.sin(state[..., 2])


@jax.jit
def speed_size(state):
    state = make_batch(state)
    return state[..., 3]


@jax.jit
def control_magnitude(action):
    action = make_batch(action)
    return np.sum(np.square(action), axis=-1)


# Numerical
@jax.jit
def index_feat(data, index):
    data = make_batch(data)
    return data[..., index]


@jax.jit
def quadratic_feat(data, goal=None):
    data = make_batch(data)
    if goal is None:
        goal = np.zeros_like(data)
    return np.square(data - goal)


@jax.jit
def abs_feat(data, goal=None):
    data = make_batch(data)
    if goal is None:
        goal = np.zeros_like(data)
    return np.abs(data - goal)


@jax.jit
def neg_feat(data):
    data = make_batch(data)
    return -1 * data


@jax.jit
def relu_feat(data):
    """ f(x) = x if x >= 0; 0 otherwise """
    data = make_batch(data)
    return (np.abs(data) + data) / 2.0


@jax.jit
def neg_relu_feat(data):
    """ f(x) = x if x < 0; 0 otherwise """
    data = make_batch(data)
    return -(np.abs(-data) - data) / 2.0


@jax.jit
def sigmoid_feat(data, mu=1.0):
    data = make_batch(data)
    return 0.5 * (np.tanh(data / (2.0 * mu)) + 1)


@jax.jit
def diff_feat(data, subtract):
    """ f(x, sub) = x - sub """
    data = make_batch(data)
    return data - subtract


@jax.jit
def neg_exp_feat(data, mu):
    """ f(x, mu) = -exp(x / mu) """
    data = make_batch(data)
    return np.exp(-data / mu)


@jax.jit
def gaussian_feat(data, sigma=None, mu=0.0):
    """ Assumes independent components"""
    data = make_batch(data)
    dim = data.shape[-1]
    # Make sigma diagonalizable vector
    if sigma is None:
        sigma = np.eye(dim)
    else:
        sigma = np.atleast_1d(sigma) ** 2

    # Number of data points
    num = data.shape[-2]
    feats = []
    diff = data - mu
    for i in range(num):
        diff_i = diff[i, :]
        exp = np.exp(-0.5 * diff_i @ np.diag(1 / sigma) @ diff_i.T)
        gaus = exp / (np.sqrt((2 * np.pi) ** dim * np.prod(sigma)))
        # gaus = np.sum(gaus, axis=-1)
        gaus = np.sum(gaus)
        feats.append(gaus)
    feats = np.array(feats)
    return feats


@jax.jit
def bounded_feat(data, lower, upper, width):
    data = make_batch(data)
    low = np.exp((lower - data) / width)
    high = np.exp((data - upper) / width)
    return low + high
