import jax.numpy as np


class Object(object):
    def __init__(self, state, name):
        self._name = name
        self._state = state

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state
