"""Feature Utilities.

Written in functional programming style such that they can
be jit-complied and run at light speed ;)

If anything isn't straightforward to follow, feel free to
contact hzyjerry@berkeley.edu

Credits:
    * Jerry Z. He 2019-2020

"""

from jax.lax import map as jmap
import jax.numpy as np
import jax

XDIM = 4
UDIM = 2
POSDIM = 2  # (x, y) coordinate


@jax.jit
def make_batch(x):
    if type(x) == list:
        x = np.asarray(x)
    if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
    return np.asarray(x)


# ====================================================
# ============ Environmental Features ================
# ====================================================


@jax.jit
def dist_to(x, y):
    """L2-norm distance dist(x, y).

    Args:
        x (ndarray): batched 2D state
        y (ndarray): target, 1D or 2D

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert y.shape[-1] == POSDIM or y.shape[-1] == XDIM

    diff = np.array(x[:, :2]) - np.array(y[..., :2])
    return np.linalg.norm(diff, axis=1, keepdims=True)


@jax.jit
def dist_to_segment(x, pt1, pt2):
    """ Distance to line segment pt1-pt2

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)
        pt1, pt2 (ndarray): target, 1D or 2D, (nbatch, 4) or (nbatch, 2)

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert len(pt1.shape) == 1 and pt1.shape[-1] == POSDIM
    assert len(pt2.shape) == 1 and pt2.shape[-1] == POSDIM

    x = x[:, :2]
    pt1 = pt1[..., :2]
    pt2 = pt2[..., :2]
    delta = pt2 - pt1
    sum_sqr = np.sum(np.square(delta))
    ui = np.sum(delta * (x - pt1) / (sum_sqr + 1e-8), axis=-1, keepdims=True)
    ui = np.clip(ui, 0, 1.0)
    dx = pt1 + ui * delta - x
    return np.linalg.norm(dx, axis=-1, keepdims=True)


@jax.jit
def diff_to(x, y):
    """ Compute x - y, requires x batch shaped.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)
        y (ndarray): target, 1D or 2D, (nbatch, 4) or (4)
    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert y.shape[-1] == POSDIM or y.shape[-1] == XDIM

    diff = np.array(x[:, :2]) - np.array(y[..., :2])
    return diff


@jax.jit
def dist_to_lane(x, center, normal):
    """Dot product of lane.normal and diff_to_lane.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)
        center, normal (ndarray): target, 1D or 2D

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert center.shape[-1] == POSDIM
    assert normal.shape[-1] == POSDIM

    diff = np.array(center) - np.array(x[:, :2])
    return np.abs(np.sum(normal * diff, axis=-1, keepdims=True))


@jax.jit
def dist_inside_fence(x, center, normal):
    """How much distance inside fence.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)
        center (array(2,)): center position, (nbatch, 2) or (2,)
        normal (array(2,)): normal direction, pointing inside lane, (nbatch, 2) or (2,)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert len(center.shape) == 1 and center.shape[-1] == POSDIM
    assert len(normal.shape) == 1 and normal.shape[-1] == POSDIM

    diff = np.array(x[:, :2]) - np.array(center)
    return np.sum(normal * diff, axis=-1, keepdims=True)


@jax.jit
def dist_outside_fence(x, center, normal):
    """How much distance outside fence.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)
        center (array(2,)): center position, (nbatch, 2) or (2,)
        normal (array(2,)): normal direction, pointing inside lane, (nbatch, 2) or (2,)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    assert len(center.shape) == 1 and center.shape[-1] == POSDIM
    assert len(normal.shape) == 1 and normal.shape[-1] == POSDIM
    return np.maximum(-1 * dist_inside_fence(x, center, normal), 0)


@jax.jit
def speed_forward(x):
    """Forward speed.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)

    """
    assert len(x.shape) == 2 and x.shape[1] == UDIM
    return x[:, 3, None] * np.sin(x[:, 2, None])


@jax.jit
def speed_size(x):
    """Forward magnitude.

    Args:
        x (ndarray): batched 2D state, (nbatch, 4)

    """
    assert len(x.shape) == 2 and x.shape[1] == XDIM
    return x[:, 3, None]


@jax.jit
def control_magnitude(u):
    """L-2 norm of control force.

    Args:
        u (ndarray): batched 2D action, (nbatch, 2)

    Output:
        out (ndarray): (nbatch, 1)

    Convention:
        * Action: [turn, accel/brake]

    """
    assert len(u.shape) == 2 and u.shape[1] == UDIM
    return np.linalg.norm(u, axis=1, keepdims=True)


@jax.jit
def control_throttle(u):
    """Acceleration force.

    Args:
        u (ndarray): batched 2D action, (nbatch, 2)

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(u.shape) == 2 and u.shape[1] == UDIM
    throttle = np.maximum(u * np.array([0, 1]), 0)
    return np.sum(throttle, axis=1, keepdims=True)


@jax.jit
def control_brake(u):
    """Brake force.

    Args:
        u (ndarray): batched 2D action, (nbatch, 2)

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(u.shape) == 2 and u.shape[1] == UDIM
    brake = np.minimum(u * np.array([0, 1]), 0)
    return np.sum(brake, axis=1, keepdims=True)


@jax.jit
def control_turn(u):
    """Turning force.

    Args:
        u (ndarray): batched 2D action, (nbatch, 2)

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert len(u.shape) == 2 and u.shape[1] == UDIM
    turn = np.abs(u * np.array([1, 0]))
    return np.sum(u, axis=1, keepdims=True)


# ====================================================
# ============== Numerical Features ==================
# ====================================================


@jax.jit
def more_than(x, y):
    """Compute x - y if more, or 0 if less.

    Args:
        x (ndarray): batched 2D state, (nbatch, dim)
        y : target, float, (nbatch, dim) or (dim,)

    """
    assert len(x.shape) == 2
    y = np.array(y)
    assert len(y.shape) == 0 or y.shape[-1] == x.shape[1]
    return np.maximum(x - y, 0)


@jax.jit
def less_than(x, y):
    """Compute y - x if less, or 0 if more.

    Args:
        x (ndarray): batched 2D state, (nbatch, dim)
        y (ndarray): target, float, (nbatch, dim) or (dim,)

    """
    assert len(x.shape) == 2
    y = np.array(y)
    assert len(y.shape) == 0 or y.shape[-1] == x.shape[1]
    return np.maximum(y - x, 0)


##==================================================
##========== Numerical features functions ==========
##==================================================


def is_state(data):
    """Wheter data look like (nbatch, dim)"""
    return len(data.shape) == 2


def is_item_state(data):
    """Whether data look like (nbatch, nitem, dim)"""
    return len(data.shape) == 3


def is_numeric(data):
    """Whether data is numerical value, e.g. float"""
    return len(np.array(data).shape) == 0


@jax.jit
def item_index_feat(data, index):
    """Compute data[:, index].

    Args:
        data (ndarray): batched 2D state, (nbatch, ndata, dim)
        index: int

    Output:
        out (ndarray): (nbatch, 1)

    """
    assert is_item_state(data)
    assert is_numeric(index)
    return data[:, index, None, :]


@jax.jit
def quadratic_feat(data, goal=None, max_val=np.inf):
    """Compute square(data - goal).

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitem, dim)
        goal (ndarray): target, (nbatch, dim) or (dim,)

    """
    assert is_state(data) or is_item_state(data)
    if goal is None:
        goal = np.zeros_like(data)
    diff_val = data - goal
    diff_val = np.minimum(diff_val, max_val)
    return np.square(diff_val)


@jax.jit
def abs_feat(data, goal=None):
    """Compute abs(data - goal).

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitems, dim)
        goal (ndarray): target, (nbatch, dim) or (dim,)

    """
    assert is_state(data) or is_item_state(data)
    if goal is None:
        goal = np.zeros_like(data)
    return np.abs(data - goal)


@jax.jit
def neg_feat(data):
    """Compute -data.

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitems, dim)

    """
    assert is_state(data) or is_item_state(data)
    return -1 * data


@jax.jit
def positive_const_feat(data):
    """Constant if greater than 0. Used to test boundaries.

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitems, dim)

    """
    assert is_state(data) or is_item_state(data)
    return np.where(data > 0, np.ones_like(data), np.zeros_like(data))


@jax.jit
def relu_feat(data, max_val=np.inf):
    """Compute f(x) = x if x >= 0; 0 otherwise.

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitems, dim)

    """
    assert is_state(data) or is_item_state(data)
    val = np.maximum(data, 0)
    return np.minimum(val, max_val)


@jax.jit
def neg_relu_feat(data):
    """Compute f(x) = x if x < 0; 0 otherwise.

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitem, dim)

    """
    assert is_state(data) or is_item_state(data)
    return np.minimum(data, 0)


@jax.jit
def sigmoid_feat(data, mu=1.0):
    """Compute sigmoid function.

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, nitem, dim)
        mu (mdarray): float or (dim,)

    """
    assert is_state(data) or is_item_state(data)
    mu = np.array(mu)
    assert len(mu.shape) == 0 or len(mu.shape) == 1 and mu.shape[0] == data.shape[1]
    return 0.5 * (np.tanh(data / (2.0 * mu)) + 1)


@jax.jit
def neg_exp_feat(data, mu):
    """Compute -exp(x / mu).

    Args:
        data (ndarray): batched 2D state, (nbatch, dim) or (nbatch, dim)
        mu (mdarray): float or (nbatch,)

    """
    assert is_state(data) or is_item_state(data)
    mu = np.array(mu)
    assert len(mu.shape) == 0 or len(mu.shape) == 1 and mu.shape[0] == data.shape[1]
    return np.exp(-data / mu)


@jax.jit
def gaussian_feat(data, sigma=None, mu=0.0):
    """Compute Normal(data, sigma, mu).

    Args:
        data (ndarray): batched 2D state, (nbatch, n_data, dim)
        sigma: int or (dim, )
        mu: int or (dim, )

    Output:
        out: (nbatch, n_data, 1)

    Note:
        Assumes independent components.

    """
    assert is_item_state(data)
    mu = np.array(mu)
    assert len(mu.shape) == 0 or len(mu.shape) == 1 and mu.shape[0] == data.shape[2]
    # Make sigma diagonalized vector
    dim = data.shape[2]
    if sigma is None:
        sigma_arr = np.eye(dim)
    else:
        sigma_arr = np.array(sigma)
        assert (
            len(sigma_arr.shape) == 0
            or len(sigma_arr.shape) == 1
            and sigma_arr.shape[0] == data.shape[2]
        )
        sigma_arr = np.atleast_1d(sigma_arr) ** 2

    # Collapse n_data (dimension 1)
    data_shape = data.shape
    data = data.reshape(-1, data.shape[2])
    feats = []
    diff = data - mu

    def dim_i_gaussian(diff_i):
        assert len(diff_i.shape) == 1
        exp = np.exp(-0.5 * diff_i @ np.diag(1 / sigma_arr) @ diff_i.T)
        gaus = exp / (np.sqrt((2 * np.pi) ** dim * np.prod(sigma_arr)))
        gaus = np.sum(gaus)
        return gaus

    # pad one dimension to output (nbatch, 1)
    feats = np.array(jmap(dim_i_gaussian, diff))[:, None]
    # Recover n_data (dimension 1)
    feats = feats.reshape((data_shape[0], data_shape[1], 1))
    return feats


@jax.jit
def exp_bounded_feat(data, lower, upper, width):
    """Exponential bounded feature.

    Used for `cost = exp(over bound)`

    Args:
        data (ndarray): batched 2D state, (nbatch, dim)

    """
    assert is_state(data)
    lower = np.array(lower)
    upper = np.array(upper)
    assert (
        len(lower.shape) == 0
        or len(lower.shape) == 1
        and lower.shape[0] == data.shape[1]
    )
    assert (
        len(upper.shape) == 0
        or len(upper.shape) == 1
        and upper.shape[0] == data.shape[1]
    )

    low = np.exp((lower - data) / width)
    high = np.exp((data - upper) / width)
    return low + high
