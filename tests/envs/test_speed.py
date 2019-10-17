import torch
import jax.numpy as np
from torch.autograd import Variable
from jax import jit, grad, random
import numpy as onp
import time

key = random.PRNGKey(0)


def test_torch():
    x = Variable(torch.zeros(1000, 1000), requires_grad=True)
    sum_x = torch.sum(x ** 2)
    now = time.time()
    for _ in range(500):
        x.data = torch.rand((1000, 1000), requires_grad=True)
        sum_x = torch.sum(x ** 2)
        sum_x.backward()
    print(f"Torch Took {time.time() - now}")


def test_jax():
    def sum_square(input):
        return np.sum(np.square(input))

    grad_sum = jit(grad(sum_square))
    now = time.time()
    for _ in range(500):
        grad_sum(random.uniform(key, (1000, 1000)))
    print(f"Jax Took {time.time() - now}")


def test_jax_chain():
    def fn1(input):
        return np.square(input)

    def fn2(input):
        return np.sum(fn1(input))

    grad_fn = jit(grad(fn2))
    now = time.time()
    for _ in range(500):
        grad_fn(random.uniform(key, (1000, 1000)))
    print(f"Jax Chain Took {time.time() - now}")
