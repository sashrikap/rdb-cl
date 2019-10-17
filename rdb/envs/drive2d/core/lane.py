import jax.numpy as np
from rdb.envs.drive2d.core import feature

"""
TODO:
[1] run by batch dist2
"""


class StraightLane(object):
    """ Straight lane defined by
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
        """ Shift in normal direction by num lanes """
        shift = self.normal * self.width * num
        pt1 = self.pt1 + shift
        pt2 = self.pt2 + shift
        return StraightLane(pt1, pt2, self.width)
