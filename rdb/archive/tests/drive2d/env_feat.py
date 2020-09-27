import jax
import jax.numpy as jnp
from rdb.envs.drive2d.core.lane import *
from rdb.envs.drive2d.core.feature import *
import gym
import rdb.envs.drive2d


def test_env_feats():
    env = gym.make("Week3_01-v0")
    obs = env.reset()
    feat_fns = env.feat_fns
    x_dim = len(env.state)
    u_dim = 2
    dist_car_fn = feat_fns["dist_car"]
    dist_lane_fn = feat_fns["dist_lane"]
    speed_fn = feat_fns["speed"]
    control_fn = feat_fns["control"]

    state = jnp.zeros(x_dim)
    action = jnp.zeros(u_dim)
    assert dist_car_fn(state, action).shape == (2, 1)
    assert dist_lane_fn(state, action).shape == (3, 1)
    assert speed_fn(state, action).shape == (1,)
    assert control_fn(state, action).shape == (1,)


def test_rew_feats():
    env = gym.make("Week3_01-v0")
    obs = env.reset()
    rew_fn = env.main_car.reward_fn
    x_dim = len(env.state)
    u_dim = 2

    state = jnp.zeros(x_dim)
    action = jnp.zeros(u_dim)
    print(rew_fn(state, action))
    assert rew_fn(state, action).shape == ()
