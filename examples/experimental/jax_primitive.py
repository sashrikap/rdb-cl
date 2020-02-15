import jax.numpy as jnp
import numpy as onp
import jax
import functools
import traceback
from jax import core
from jax import abstract_arrays

# Now we register the XLA compilation rule with JAX
# TODO: for GPU? and TPU?
from jax import xla


_indentation = 0


def _trace(msg=None):
    """Print a message at current indentation."""
    if msg is not None:
        print("  " * _indentation + msg)


def _trace_indent(msg=None):
    """Print a message and then indent the rest."""
    global _indentation
    _trace(msg)
    _indentation = 1 + _indentation


def _trace_unindent(msg=None):
    """Unindent then print a message."""
    global _indentation
    _indentation = _indentation - 1
    _trace(msg)


def trace(name):
    """A decorator for functions to trace arguments and results."""

    def trace_func(func):  # pylint: disable=missing-docstring
        def pp(v):
            """Print certain values more succinctly"""
            vtype = str(type(v))
            if "jax.lib.xla_bridge._JaxComputationBuilder" in vtype:
                return "<JaxComputationBuilder>"
            elif "jaxlib.xla_extension.XlaOp" in vtype:
                return "<XlaOp at 0x{:x}>".format(id(v))
            elif (
                "partial_eval.JaxprTracer" in vtype
                or "batching.BatchTracer" in vtype
                or "ad.JVPTracer" in vtype
            ):
                return "Traced<{}>".format(v.aval)
            elif isinstance(v, tuple):
                return "({})".format(pp_values(v))
            else:
                return str(v)

        def pp_values(args):
            return ", ".join([pp(arg) for arg in args])

        @functools.wraps(func)
        def func_wrapper(*args):
            _trace_indent("call {}({})".format(name, pp_values(args)))
            res = func(*args)
            _trace_unindent("|<- {} = {}".format(name, pp(res)))
            return res

        return func_wrapper

    return trace_func


class expectNotImplementedError(object):
    """Context manager to check for NotImplementedError."""

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        global _indentation
        _indentation = 0
        if type is NotImplementedError:
            print("\nFound expected exception:")
            traceback.print_exc(limit=3)
            return True
        elif type is None:  # No exception
            assert False, "Expected NotImplementedError"
        else:
            return False


@trace("multiply_add_numpy")
def multiply_add_numpy(x, y, z):
    return jnp.add(jnp.multiply(x, y), z)


@trace("square_add_numpy")
def square_add_numpy(a, b):
    return multiply_add_numpy(a, a, b)


multiply_add_p = core.Primitive("multiply_add")  # Create the primitive


@trace("multiply_add_prim")
def multiply_add_prim(x, y, z):
    """The JAX-traceable way to use the JAX primitive.

    Note that the traced arguments must be passed as positional arguments
    to `bind`.
    """
    return multiply_add_p.bind(x, y, z)


@trace("square_add_prim")
def square_add_prim(a, b):
    """A square-add function implemented using the new JAX-primitive."""
    return multiply_add_prim(a, a, b)


@trace("multiply_add_impl")
def multiply_add_impl(x, y, z):
    """Concrete implementation of the primitive.

    This function does not need to be JAX traceable.
    Args:
    x, y, z: the concrete arguments of the primitive. Will only be caled with
      concrete values.
    Returns:
    the concrete result of the primitive.
    """
    # Note that we can use the original numpy, which is not JAX traceable
    return onp.add(onp.multiply(x, y), z)


# Now we register the primal implementation with JAX
multiply_add_p.def_impl(multiply_add_impl)


@trace("multiply_add_abstract_eval")
def multiply_add_abstract_eval(xs, ys, zs):
    """Abstract evaluation of the primitive.

  This function does not need to be JAX traceable. It will be invoked with
  abstractions of the actual arguments.
  Args:
    xs, ys, zs: abstractions of the arguments.
  Result:
    a ShapedArray for the result of the primitive.
  """
    assert xs.shape == ys.shape
    assert xs.shape == zs.shape
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


# Now we register the abstract evaluation with JAX
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)


@trace("multiply_add_xla_translation")
def multiply_add_xla_translation(c, xc, yc, zc):
    """The compilation to XLA of the primitive.

  Given an XlaBuilder and XlaOps for each argument, return the XlaOp for the
  result of the function.

  Does not need to be a JAX-traceable function.
  """
    return c.Add(c.Mul(xc, yc), zc)


xla.backend_specific_translations["cpu"][multiply_add_p] = multiply_add_xla_translation

jax.jit(square_add_prim)(2.0, 10.0)
jax.jit(multiply_add_prim)(2.0, 8.0, 10.0)
