## Enable float64 for scipy optimizer
from jax.config import config

config.update("jax_enable_x64", True)
