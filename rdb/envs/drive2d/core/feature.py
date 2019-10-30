from rdb.mdp.feature import Feature
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
    if len(state.shape) == 1:
        state = np.expand_dims(state, 0)
    return np.asarray(state)


# Environment specific
@jax.jit
def dist_to(x, y):
    x = make_batch(x)
    y = make_batch(y)
    diff = np.array(x[..., :2]) - np.array(y[..., :2])
    return np.linalg.norm(diff, axis=-1)


@jax.jit
def diff_to(x, y):
    """ (x2 - x1, y2 - y1)"""
    x = make_batch(x)
    y = make_batch(y)
    diff = np.array(x[..., :2]) - np.array(y[..., :2])
    return diff


@jax.jit
def dist_to_lane(center, normal, x):
    x = make_batch(x)
    diff = np.array(center) - np.array(x[..., :2])
    return np.abs(np.sum(normal * diff, axis=-1))


@jax.jit
def dist_inside_fence(center, normal, x):
    x = make_batch(x)
    diff = np.array(x[..., :2]) - np.array(center)
    return np.sum(normal * diff, axis=-1)


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
    data = make_batch(data)
    return (np.abs(data) + data) / 2.0


@jax.jit
def neg_relu_feat(data):
    return relu_feat(neg_feat(data))


@jax.jit
def sigmoid_feat(data, mu=1.0):
    data = make_batch(data)
    return 0.5 * (np.tanh(data / (2.0 * mu)) + 1)


@jax.jit
def diff_feat(data, subtract):
    data = make_batch(data)
    return data - subtract


@jax.jit
def neg_exp_feat(data, mu):
    data = make_batch(data)
    # print(f"neg exp {np.sum(np.exp(-data / (2 * sigma ** 2)), axis=-1)}")
    return np.exp(-data / mu)


@jax.jit
def gaussian_feat(data, sigma=1.0, mu=0.0):
    quad = quadratic_feat(data, goal=mu)
    gaus = neg_exp_feat(quad, 2 * sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)
    return gaus


@jax.jit
def bounded_feat(data, lower, upper, width):
    data = make_batch(data)
    low = np.exp((lower - data) / width)
    high = np.exp((data - upper) / width)
    return low + high
