"""Object Class.

Broadly contains anything that are displayed as static on the road

Includes:
    * General object
    * Exit/Entrance for Driveway

"""

import jax.numpy as jnp
import numpy as onp
import pyglet

from rdb.envs.drive2d.core.utils import centered_image
from rdb.envs.drive2d.core.lane import StraightLane
from rdb.envs.drive2d.core.dynamics import build_identity_dynamics
from pyglet import gl, graphics
from copy import deepcopy

DEFAULT_SCALE = 0.15 / 600.0


def object_sprite(name, scale=1.0, group=None, batch=None):
    sprite = pyglet.sprite.Sprite(
        centered_image("{}.png".format(name)), subpixel=True, group=group, batch=batch
    )
    sprite.scale = scale * DEFAULT_SCALE
    # import pdb; pdb.set_trace()
    return sprite


class Object(object):
    """Object class

    Properties:
        state: (nbatch, udim), by default nbatch = 1
    """

    def __init__(self, state, name, scale=1.0, opacity=255):
        self._name = name
        if len(state.shape) == 1:
            self._state = state[None, :]
        else:
            self._state = state
        self._sprite = None
        self._scale = scale
        self._opacity = opacity
        self._xdim = 2
        self._udim = 2
        self.dynamics_fn = build_identity_dynamics(self._xdim, self._udim)

    @property
    def xdim(self):
        return self._xdim

    @property
    def udim(self):
        return self._udim

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    def copy(self):
        return Object(deepcopy(self._state), self._name, self._scale, self._opacity)

    @state.setter
    def state(self, state):
        assert len(state.shape) == 2
        self._state = state
        if self._sprite is not None:
            self._sprite.x = state[0, 0]
            self._sprite.y = state[0, 1]
            rotation = 0.0
            if self.xdim >= 3:
                rotation = state[0, 2]
            self._sprite.rotation = rotation
            self._sprite.opacity = self._opacity

    def register(self, batch, group):
        self._sprite = object_sprite(self.name, self._scale, batch=batch, group=group)
        state = self._state
        self._sprite.x = state[0, 0]
        self._sprite.y = state[0, 1]
        rotation = 0.0
        if self.xdim >= 3:
            rotation = state[0, 2]
        self._sprite.rotation = rotation
        self._sprite.opacity = self._opacity


class Obstacle(Object):
    NAME = "cone"

    def __init__(self, state, scale=0.7):
        super().__init__(state, self.NAME, scale)


class Tree(Object):
    NAME = "tree"

    def __init__(self, state, scale=0.2):
        super().__init__(state, self.NAME, scale)

class Debris(Object):
    NAME = "debris"

    def __init__(self, state, scale=0.2):
        super().__init__(state, self.NAME, scale)

class Garage(Object):
    # NAME = "Garage"
    NAME = "parking-lot3"
    # NAME = "cone"

    def __init__(self, state, scale=1.0):
        super().__init__(state, self.NAME, scale)
        # Initialize sprite
        super().render()

    def render_floor(self):
        gl.glColor3f(0.4, 0.4, 0.4)
        dx = jnp.array([1.0, 0.0])
        dy = jnp.array([0.0, 1.0])
        xlen = self._sprite.width
        ylen = self._sprite.height
        pos = jnp.array(self.state[:2])
        quad_strip = onp.hstack(
            [
                pos - dx * 0.5 * xlen - dy * 0.5 * ylen,
                pos - dx * 0.5 * xlen + dy * 0.5 * ylen,
                pos + dx * 0.5 * xlen - dy * 0.5 * ylen,
                pos + dx * 0.5 * xlen + dy * 0.5 * ylen,
            ]
        )
        graphics.draw(4, gl.GL_QUAD_STRIP, ("v2f", quad_strip))
        gl.glColor3f(1.0, 1.0, 1.0)
        line_strip = onp.hstack(
            [
                pos - 0.5 * dy * ylen - 0.5 * xlen * dx,
                pos - 0.5 * dy * ylen + 0.5 * xlen * dx,
                pos + 0.5 * dy * ylen - 0.5 * xlen * dx,
                pos + 0.5 * dy * ylen + 0.5 * xlen * dx,
                pos - 0.5 * dy * ylen - 0.5 * xlen * dx,
                pos + 0.5 * dy * ylen - 0.5 * xlen * dx,
            ]
        )
        # import pdb; pdb.set_trace()
        graphics.draw(6, gl.GL_LINES, ("v2f", line_strip))

    # def render(self):
    #     self.render_floor()
    #     # Render building
    #     super().render()


class Driveway(Object, StraightLane):
    def __init__(self, state, name, pt1, pt2, width, length):
        self._width = width
        self._length = length
        Object.__init__(self, state, name)
        StraightLane.__init__(self, pt1, pt2, width)


class Entrance(Driveway):
    NAME = "Entrance"

    def __init__(self, state, pt1, pt2, width, length):
        """Entrance Constructor.

        Attributes:
            state (nd.array(2)): the point where entrance connects with road, angle and speed (0)
            start (nd.array(2)): state[:2]
            end (nd.array(2)): the innermost point of entrance

        """
        super().__init__(state, self.NAME, pt1, pt2, width, length)
        self._start = jnp.array(self.state[:2])
        self._end = self._start - self.forward * self._length

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    # def render(self, opacity=255):
    #     gl.glColor3f(0.4, 0.4, 0.4)
    #     normal, forward, width = self.normal, self.forward, self.width
    #     quad_strip = onp.hstack(
    #         [
    #             self._start - forward * self._length - 0.5 * width * normal,
    #             self._start - forward * self._length + 0.5 * width * normal,
    #             self._start - 0.5 * width * normal,
    #             self._start + 0.5 * width * normal,
    #         ]
    #     )
    #     graphics.draw(4, gl.GL_QUAD_STRIP, ("v2f", quad_strip))
    #     gl.glColor3f(1.0, 1.0, 1.0)
    #     line_strip = onp.hstack(
    #         [
    #             self._start - forward * self._length - 0.5 * width * normal,
    #             self._start - 0.5 * width * normal,
    #             self._start - forward * self._length + 0.5 * width * normal,
    #             self._start + 0.5 * width * normal,
    #         ]
    #     )
    #     graphics.draw(4, gl.GL_LINES, ("v2f", line_strip))
