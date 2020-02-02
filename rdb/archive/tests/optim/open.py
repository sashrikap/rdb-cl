from rdb.optim.open import LocalOptimizer
import jax.numpy as np


def test_easy_u():
    def f_dyn(x, u):
        return x + u

    def f_rew(x, u, next_x):
        return -np.sum((x - 1) ** 2)

    optim = LocalOptimizer(f_dyn, f_rew, 1, 1, 5)
    opt_u, opt_r, _ = optim.optimize_u(np.zeros(1), np.zeros(5))
    print(f"Optimal control {opt_u}")
    print(f"Optimal reward {opt_r}")
    sol_u = np.array([[1], [0], [0], [0], [0]])
    assert np.sum(np.abs(opt_u - sol_u)) <= 1e-3


def test_easy_x():
    import time

    def f_dyn(x, u):
        return x + u

    def f_rew(x, u, next_x):
        return -np.sum(x ** 2)

    optim = LocalOptimizer(f_dyn, f_rew, 1, 1, 5)
    t1 = time.time()
    opt_x, opt_r, _ = optim.optimize_x0(np.ones(1), np.zeros(5))
    t2 = time.time()
    print(f"Optimal x {opt_x}")
    print(f"Optimal reward {opt_r}")
    sol_x = np.array([0])
    assert np.sum(np.abs(opt_x - sol_x)) <= 1e-3

    optim = LocalOptimizer(f_dyn, f_rew, 1, 1, 5, jit=True)
    t1_jit = time.time()
    opt_x, opt_r, _ = optim.optimize_x0(np.ones(1), np.zeros(5))
    t2_jit = time.time()
    print(f"Time no jit {t2 - t1:.3f} jit {t2_jit - t1_jit:.3f}")
    print(f"JIT speed up {(t2 - t1) / (t2_jit - t1_jit):.3f}x")
