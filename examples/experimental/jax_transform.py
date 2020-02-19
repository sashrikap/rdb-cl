import jax
import jax.numpy as np
import numpy as onp


@jax.custom_transforms
def f(x):
    return onp.array(x) * x


jax.defvjp(f, lambda g, ans, x: g)


def potential_fn(x):
    return f(x)


def test_potential():
    grad_potential = jax.grad(potential_fn)
    x = np.ones((10, 10))
    print(grad_potential(x))
    print(jax.jit(f)(x))


if __name__ == "__main__":
    test_potential()
