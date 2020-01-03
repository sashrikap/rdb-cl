"""Object Class.

Broadly contains anything that are displayed as static on the road

Includes:
    * General object
    * Exit/Entrance for Driveway

"""

import jax.numpy as np
import numpy as onp
import pyglet
from rdb.envs.drive2d.core.car import centered_image
from rdb.envs.drive2d.core.lane import StraightLane
from rdb.envs.drive2d.core.dynamics import identity_dynamics_fn
from pyglet import gl, graphics


DEFAULT_SCALE = 0.15 / 600.0


def object_sprite(name, scale=1.0):
    sprite = pyglet.sprite.Sprite(centered_image("{}.png".format(name)), subpixel=True)
    sprite.scale = scale * DEFAULT_SCALE
    return sprite


class Object(object):
    def __init__(self, state, name, scale=1.0, opacity=255):
        self._name = name
        self._state = state
        self._sprite = None
        self._scale = scale
        self._opacity = opacity
        self.dynamics_fn = identity_dynamics_fn()

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    def render(self):
        if self._sprite is None:
            self._sprite = object_sprite(self.name, self._scale)
        # self._sprite.x = self.state[1]
        # self._sprite.y = self.state[0]
        self._sprite.x = self.state[0]
        self._sprite.y = self.state[1]
        rotation = 0.0
        if len(self.state) >= 3:
            rotation = self.state[2]
        self._sprite.rotation = rotation
        self._sprite.opacity = self._opacity
        self._sprite.draw()


class Obstacle(Object):
    NAME = "cone"

    def __init__(self, state, scale=1.0):
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
        dx = np.array([1.0, 0.0])
        dy = np.array([0.0, 1.0])
        xlen = self._sprite.width
        ylen = self._sprite.height
        pos = np.array(self.state[:2])
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

    def render(self):
        self.render_floor()
        # Render building
        super().render()


class Driveway(Object, StraightLane):
    def __init__(self, state, name, pt1, pt2, width, length):
        self._width = width
        self._length = length
        Object.__init__(self, state, name)
        StraightLane.__init__(self, pt1, pt2, width)

    def render(self, opacity=255):
        pass


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
        self._start = np.array(self.state[:2])
        self._end = self._start - self.forward * self._length

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def render(self, opacity=255):
        gl.glColor3f(0.4, 0.4, 0.4)
        normal, forward, width = self.normal, self.forward, self.width
        quad_strip = onp.hstack(
            [
                self._start - forward * self._length - 0.5 * width * normal,
                self._start - forward * self._length + 0.5 * width * normal,
                self._start - 0.5 * width * normal,
                self._start + 0.5 * width * normal,
            ]
        )
        graphics.draw(4, gl.GL_QUAD_STRIP, ("v2f", quad_strip))
        gl.glColor3f(1.0, 1.0, 1.0)
        line_strip = onp.hstack(
            [
                self._start - forward * self._length - 0.5 * width * normal,
                self._start - 0.5 * width * normal,
                self._start - forward * self._length + 0.5 * width * normal,
                self._start + 0.5 * width * normal,
            ]
        )
        graphics.draw(4, gl.GL_LINES, ("v2f", line_strip))
