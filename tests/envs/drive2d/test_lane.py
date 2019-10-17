from rdb.envs.drive2d.core.lane import *
import jax.numpy as np


def test_shift_dist():
    lane = StraightLane([1.0, 0], [-1.0, 0], 0.4)
    assert np.allclose(lane.pt1, [1.0, 0])
    assert np.allclose(lane.normal, [0, 1.0])
    assert np.allclose(lane.forward, [1.0, 0.0])

    lane2 = lane.shifted(1)
    assert np.allclose(lane2.pt1, [1.0, 0.4])
    assert np.allclose(lane2.pt2, [-1.0, 0.4])
