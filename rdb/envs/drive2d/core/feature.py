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
def dist2(x, y):
    x = make_batch(x)
    y = make_batch(y)
    diff = np.array(x[..., :2]) - np.array(y[..., :2])
    return np.linalg.norm(diff, axis=-1)


@jax.jit
def dist2lane(center, normal, x):
    x = make_batch(x)
    diff = np.array(center) - np.array(x[..., :2])
    return np.abs(np.sum(normal * diff, axis=-1))


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
    return np.sum(np.square(action ** 2), axis=-1)


# Numerical
@jax.jit
def index_feat(data, index):
    data = make_batch(data)
    # print("index data", data)
    return data[..., index]


@jax.jit
def quadratic_feat(data, goal=None):
    if goal is None:
        goal = np.zeros_like(data)
    return np.sum(np.square(data - goal), axis=-1)


@jax.jit
def abs_feat(data, goal=None):
    if goal is None:
        goal = np.zeros_like(data)
    return np.sum(np.square(data - goal), axis=-1)


@jax.jit
def exp_feat(data, sigma):
    return np.sum(np.exp(-data / sigma ** 2), axis=-1)


@jax.jit
def bounded_feat(data, lower, upper, width):
    low = np.exp((lower - data) / width)
    high = np.exp((data - upper) / width)
    return np.sum(low + high)
