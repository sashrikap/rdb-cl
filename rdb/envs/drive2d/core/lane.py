"""Lane object.

TODO:
[1] run by batch dist2
"""

import jax.numpy as np
import numpy as onp
from rdb.envs.drive2d.core import feature
from pyglet import gl, graphics


class StraightLane(object):
    """Straight lane class.

    Utility functions for optimization

    Note:
    * Straight lane defined by
         Forward
      |     |     |
      +pt1  |     +pt2
      |     |     |
         Backward

    """

    def __init__(self, pt1, pt2, width):
        self.pt1 = np.asarray(pt1)
        self.pt2 = np.asarray(pt2)
        self.center = (self.pt1 + self.pt2) / 2
        self.width = width
        dist = np.linalg.norm(self.pt1 - self.pt2)
        self.forward = (self.pt1 - self.pt2) / dist
        self.normal = np.array([-self.forward[1], self.forward[0]])

    def shifted(self, num):
        """Shift in normal direction by num lanes.
        """

        shift = self.normal * self.width * num
        pt1 = self.pt1 + shift
        pt2 = self.pt2 + shift
        return StraightLane(pt1, pt2, self.width)

    def register(self, batch, group):
        normal, forward = self.normal, self.forward
        pt1, pt2, width = self.pt1, self.pt2, self.width
        W = 10
        quad_strip = onp.hstack(
            [
                pt1 - forward * W - 0.5 * width * normal,
                pt1 - forward * W + 0.5 * width * normal,
                pt2 + forward * W - 0.5 * width * normal,
                pt2 + forward * W + 0.5 * width * normal,
            ]
        )
        colors = [int(0.4 * 255)] * 12
        batch.add(4, gl.GL_QUAD_STRIP, group, ("v2f", quad_strip), ("c3B", colors))
        line_strip = onp.hstack(
            [
                pt1 - forward * W - 0.5 * width * normal,
                pt1 + forward * W - 0.5 * width * normal,
                pt1 - forward * W + 0.5 * width * normal,
                pt1 + forward * W + 0.5 * width * normal,
            ]
        )
        colors = [255] * 12
        batch.add(4, gl.GL_LINES, group, ("v2f", line_strip), ("c3B", colors))
