import jax.numpy as np
import jax


def test_multi_arguments():
    def func(arg1, arg2):
        return np.sum(arg1 * arg2)

    grad1 = jax.grad(func)
    args = (np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    result = np.array([3.0, 4.0])
    assert np.allclose(grad1(*args), result)


def test_index():
    """ Test some indexing operations and whether JAX preserves gradient"""

    def loss_sum(vals):
        loss = np.sum(vals[:-1])
        vals = vals[1:]
        loss += np.sum(vals)
        return loss

    grad_sum = jax.grad(loss_sum)
    gs = grad_sum(np.array([0, 1, 2], dtype=float))
    assert np.allclose(gs, np.array([1, 2, 1], dtype=float))


def test_multivariable():
    def loss_sum(val):
        loss = np.sum(val[:2] ** 2 + val[2:] ** 3)
        return loss

    grad = jax.grad(loss_sum)
    gs = grad(np.array([2, 3, 4, 5], dtype=float))


def test_retain():
    import time

    a = 10

    def loss_sum(val):
        loss = np.sum(val[:2] ** 2 + val[2:] ** 3) * a
        return loss

    grad = jax.grad(loss_sum)
    gs = grad(np.array([2, 3, 4, 5], dtype=float))
    assert np.allclose(gs, np.array([40.0, 60.0, 480.0, 750.0]))
    a = 20
    t1 = time.time()
    gs = grad(np.array([2, 3, 4, 5], dtype=float))
    t2 = time.time()
    _grad = jax.jit(grad)
    _gs = _grad(np.array([2, 3, 4, 5], dtype=float))
    t3 = time.time()

    dt1 = t2 - t1
    dt2 = t3 - t2
    assert np.allclose(gs, np.array([80.0, 120.0, 960.0, 1500.0]))


def test_class():
    class optimizer(object):
        def __init__(self, x0):
            self.x0 = x0
            self.grad_u = jax.grad(self.loss)

        def loss(self, u):
            return np.sum((u - self.x0) ** 2)

    optim = optimizer(np.array([1.0, 2.0]))
    gs = optim.grad_u(np.array([2.0, 3.0]))
    assert np.allclose(gs, np.array([2.0, 2.0]))
    optim.x0 = np.array([2.0, 2.0])
    gs = optim.grad_u(np.array([2.0, 3.0]))
    assert np.allclose(gs, np.array([0.0, 2.0]))


def test_args():
    def loss_sum(val, arg):
        loss = np.sum(val[:2] ** 2 + val[2:] ** 3) * arg
        return loss

    grad = jax.grad(loss_sum)
    gs = grad(np.array([2, 3, 4, 5], dtype=float), 3)
    assert np.allclose(gs, np.array([12.0, 18.0, 144.0, 225.0]))


def test_juxt():
    from toolz.functoolz import juxt, compose
    import jax, time

    key = jax.random.PRNGKey(0)

    funcs = [np.sum] * 100

    def juxt_max1(*args):
        return np.max([fn(*args) for fn in funcs])

    juxt_max2 = compose(np.max, juxt(funcs))
    now = time.time()
    grad1 = jax.jit(jax.grad(juxt_max1))
    print("t1", time.time() - now)
    now = time.time()
    grad2 = jax.jit(jax.grad(juxt_max1))
    print("t2", time.time() - now)
    now = time.time()
    grad1(jax.random.uniform(key, (100, 100))).shape
    print("t3", time.time() - now)
    now = time.time()
    grad2(jax.random.uniform(key, (100, 100))).shape
    print("t4", time.time() - now)
    now = time.time()
