from rdb.envs.drive2d.core.constraints import *
from functools import partial
import jax.numpy as jnp
import gym


def test_overspeed():
    env = gym.make("Week3_02-v0")
    # main_car driving at 0.7
    env.reset()
    action = jnp.zeros(2)
    constr_overspeed = partial(is_overspeed, env=env, max_speed=0.5)
    assert jnp.all(constr_overspeed(env.state, action))

    constr_overspeed = partial(is_overspeed, env=env, max_speed=0.8)
    assert jnp.all(not constr_overspeed(env.state, action))


def test_underspeed():
    env = gym.make("Week3_02-v0")
    # main_car driving at 0.7
    env.reset()
    action = jnp.zeros(2)
    constr_underspeed = partial(is_underspeed, env=env, min_speed=0.5)
    assert jnp.all(not constr_underspeed(env.state, action))

    constr_underspeed = partial(is_underspeed, env=env, min_speed=0.8)
    assert jnp.all(constr_underspeed(env.state, action))


def test_wronglane():
    env = gym.make("Week3_02-v0")
    # main_car driving at 0.7
    env.reset()
    action = jnp.zeros(2)
    constr_wronglane = partial(is_wronglane, env=env, lane_idx=0)
    assert jnp.all(not constr_wronglane(env.state, action))

    constr_wronglane = partial(is_wronglane, env=env, lane_idx=1)
    assert jnp.all(constr_wronglane(env.state, action))

    constr_wronglane = partial(is_wronglane, env=env, lane_idx=0)
    assert jnp.all(not constr_wronglane(env.state, action))


def test_uncomfortable():
    env = gym.make("Week3_02-v0")
    env.reset()
    max_actions = jnp.array([1.0, 1.0])
    constr_uncomfortable = partial(is_uncomfortable, env=env, max_actions=max_actions)
    assert jnp.all(not constr_uncomfortable(env.state, jnp.zeros(2)))
    assert jnp.all(constr_uncomfortable(env.state, jnp.ones(2) * 2.0))


def test_collision():
    RENDER = False
    env = gym.make("Week3_02-v0")

    # main_car driving at 0.7
    env.reset()
    import copy

    state_front = copy.deepcopy(env.state)
    state_front[0] = state_front[4] = state_front[8] = 0.0
    state_front[9] = state_front[5] + env.car_length
    state_side = copy.deepcopy(env.state)
    state_side[0] = state_side[4]
    state_side[9] = state_side[5]
    state_side[8] = state_side[4] + env.car_width

    def render_state(state):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        env.set_init_state(state)
        env.reset()
        env.step(jnp.zeros(2))
        fig = plt.figure()
        img = env.render("rgb_array")
        plt.imshow(img)
        plt.show()

    if RENDER:
        # Render front collision
        render_state(state_front)
        # Render side collision
        render_state(state_side)

    env.set_init_state(state_front)
    env.reset()
    constr_collision = partial(is_collision, env=env)
    assert jnp.all(not constr_collision(state_front, jnp.zeros(2)))
    state_front[9] -= 0.1
    env.set_init_state(state_front)
    env.reset()
    assert jnp.all(constr_collision(state_front, jnp.zeros(2)))

    env.set_init_state(state_side)
    env.reset()
    constr_collision = partial(is_collision, env=env)
    assert jnp.all(not constr_collision(state_side, jnp.zeros(2)))
    state_side[8] -= 0.1
    env.set_init_state(state_side)
    env.reset()
    assert jnp.all(constr_collision(state_side, jnp.zeros(2)))


def test_offtrack():
    RENDER = False
    env = gym.make("Week3_02-v0")

    # main_car driving at 0.7
    env.reset()
    import copy

    state_left = copy.deepcopy(env.state)
    state_left[8] = -1.5 * env.lane_width + 0.5 * env.car_width + 0.05
    state_right = copy.deepcopy(env.state)
    state_right[8] = 1.5 * env.lane_width - 0.5 * env.car_width - 0.05

    def render_state(state):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        env.set_init_state(state)
        env.reset()
        env.step(jnp.zeros(2))
        fig = plt.figure()
        img = env.render("rgb_array")
        plt.imshow(img)
        plt.show()

    if RENDER:
        # Render left collision
        render_state(state_left)
        # Render right collision
        render_state(state_right)

    env.set_init_state(state_left)
    env.reset()
    constr_offtrack = partial(is_offtrack, env=env)
    assert jnp.all(not constr_offtrack(state_left, jnp.zeros(2)))
    state_left[8] -= 0.1
    env.set_init_state(state_left)
    env.reset()
    assert jnp.all(constr_offtrack(state_left, jnp.zeros(2)))

    env.set_init_state(state_right)
    env.reset()
    constr_offtrack = partial(is_offtrack, env=env)
    assert jnp.all(not constr_offtrack(state_right, jnp.zeros(2)))
    state_right[8] += 0.1
    env.set_init_state(state_right)
    env.reset()
    assert jnp.all(constr_offtrack(state_right, jnp.zeros(2)))
