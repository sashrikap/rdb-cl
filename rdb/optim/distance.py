import jax
import jax.numpy as np
from rdb.envs.drive2d.core.feature import gaussian_feat, make_batch

"""
Trajectory distance estimation methods

Includes:
[1] Kernel Density Estimator

"""


def kernel_distance_fn(trajs, kernel="Gaussian"):
    """ Kernel distance function for trajectory

    Params
    : trajs : list(trajectories)
    """
    assert False, "Deprecated, please use `scipy.stats.gaussian_kde"
    std = np.std(trajs, axis=0)
    eta = 1e-6

    @jax.jit
    def gaussian_distance_fn(traj1, traj2):
        """
        Params
        : traj1, traj2 : (T, s_dim)
        """
        assert len(traj1.shape) == 2 and len(traj2.shape) == 2
        traj1 = make_batch(traj1)
        traj2 = make_batch(traj2)
        import pdb

        pdb.set_trace()
        const = 1 / np.sum(gaussian_feat(np.zeros_like(traj1), sigma=(std + eta)))
        gauss = np.sum(gaussian_feat((traj1 - traj2), sigma=(std + eta)))
        return const * gauss

    if kernel == "Gaussian":
        return gaussian_distance_fn
    else:
        raise NotImplementedError
