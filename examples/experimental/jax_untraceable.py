import jax
import jax.numpy as np
from jax import lax, random
import numpy as onp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import abstract_arrays
from jax.interpreters import xla, ad
from numpyro.util import control_flow_prims_disabled, fori_loop, optional

# @jax.jit
def experimental_fn(x):
    data = onp.random.random((2, 2))
    # data = onp.array(x)
    print("data", data)
    result = np.sum(x * data)
    return result


def test_one():
    grad_experimental_fn = jax.grad(experimental_fn)
    for _ in range(5):
        g1 = grad_experimental_fn(np.ones(5))
        print(g1)
        f1 = experimental_fn(np.ones(5))
        print(f1)


untraceable_primitive_p = jax.core.Primitive("untraceable_primitive")
untraceable_primitive_p.multiple_results = True


def untraceable_primitive(x):
    return untraceable_primitive_p.bind(x)


def untraceable_primitive_impl(x):
    # we don't want jax to trace this function: onp.array
    return (np.array(x),)


untraceable_primitive_p.def_impl(untraceable_primitive_impl)


# untraceable_primitive_p.def_abstract_eval(lambda x: x)
untraceable_primitive_p.def_abstract_eval(
    lambda a: (abstract_arrays.raise_to_shaped(a),)
)


xla.translations[untraceable_primitive_p] = xla.lower_fun(
    untraceable_primitive_impl, instantiate=True
)

# ad.defjvp2(untraceable_primitive_p, None, lambda tangent, ans, key, a: (np.ones_like(ans[0]),))
ad.defvjp(untraceable_primitive_p, lambda g, x: g * 0)


def experimental_two(x):
    data = untraceable_primitive(x)
    result = x.dot(data[0])
    return result


def test_two():
    grad_experimental_fn = jax.grad(experimental_two)
    x = np.ones(5) * 5
    print(experimental_two(x))
    print(jax.jit(experimental_two)(x))
    g2 = grad_experimental_fn(x)
    print(g2)


def test_kernel():
    # with jax.disable_jit(), control_flow_prims_disabled():
    dist_cars = numpyro.sample("prior_data", dist.Uniform(-10, 10), sample_shape=(2, 2))
    # rand = onp.random.random((2, 2)) + np.array(dist_cars)
    rand = np.array(dist_cars)
    # rand = untraceable_primitive(dist_cars)
    print("rand", rand)
    log_prob = np.sum(rand * dist_cars)
    numpyro.factor("forward_log_prob", log_prob)


def test_numpyro():
    # with jax.disable_jit(), control_flow_prims_disabled():
    rng_key = random.PRNGKey(2)
    kernel = NUTS(test_kernel)
    num_warmup = 20
    num_samples = 40
    mcmc = MCMC(kernel, 20, 40, progress_bar=True)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()


if __name__ == "__main__":
    # test_one()
    # test_numpyro()
    test_two()
    # test_kernel()
