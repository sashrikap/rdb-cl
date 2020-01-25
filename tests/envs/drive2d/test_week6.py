import gym
import rdb.envs.drive2d
import jax.numpy as np


def test_natural():
    tasks = [(0.4, -0.4, -0.13, 0.4)]
    env = gym.make("Week6_01-v1")
    env.reset()
    natural = [True]
    natural_tasks = np.array(tasks)[np.array(natural)]
    env_tasks = env._get_natural_tasks(tasks)
    import pdb

    pdb.set_trace()
    assert len(env_tasks) == len(natural_tasks) and np.allclose(
        env_tasks, natural_tasks
    )
