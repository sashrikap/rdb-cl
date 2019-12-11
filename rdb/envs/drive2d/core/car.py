import jax.numpy as np
import jax
import pyglet
import numpy as onp
from rdb.envs.drive2d.core.dynamics import *
from rdb.optim.utils import *
from copy import deepcopy


def centered_image(filename):
    img = pyglet.resource.image(filename)
    img.anchor_x = img.width / 2.0
    img.anchor_y = img.height / 2.0
    return img


def car_sprite(color, scale=0.15 / 600.0):
    sprite = pyglet.sprite.Sprite(
        centered_image("car-{}.png".format(color)), subpixel=True
    )
    sprite.scale = scale
    return sprite


class Car(object):
    def __init__(self, env, init_state, horizon, color, friction=0.1):
        """General Car Object.

        Args:
            init_state: initial state
            horizon: planning horizon
        """
        self.env = env
        self.horizon = horizon
        self.dynamics_fn = car_dynamics_fn(friction)
        self._init_state = init_state
        self._state = init_state
        self._sprite = None
        self._color = color
        # Initilialize trajectory data

    def reset(self):
        self._state = self._init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, init_state):
        self._init_state = init_state

    @property
    def color(self):
        return self._color

    def control(self, u, dt):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def render(self, opacity=255):
        if self._sprite is None:
            self._sprite = car_sprite(self.color)
        self._sprite.x, self._sprite.y = self.state[0], self.state[1]
        # sprite.x, sprite.y = 0, 0
        self._sprite.rotation = -self.state[2] * 180 / onp.pi
        self._sprite.opacity = opacity
        self._sprite.draw()


class FixSpeedCar(Car):
    def __init__(self, env, init_state, fix_speed, horizon=1, color="red"):
        self.env = env
        super().__init__(env, init_state, horizon, color)
        self.fix_speed = fix_speed
        self.dynamics_fn = fixspeed_dynamics_fn(fix_speed)

    def control(self, dt):
        self._state += self.dynamics_fn(self._state, None) * dt

    def copy(self):
        return FixSpeedCar(
            self.env,
            deepcopy(self._init_state),
            self.fix_speed,
            self.horizon,
            self.color,
        )


class OptimalControlCar(Car):
    def __init__(self, env, cost_weights, init_state, horizon=10, color="yellow"):
        """Autonomous car with optimal controller.

        Args:
            env: world
            cost_weights (list): keys implicit in env.feature_keys

        """
        super().__init__(env, init_state, horizon, color)
        self._cost_weights = cost_weights
        self._features_fn = None
        self._cost_fn = None
        self._cost_runtime = None

    @property
    def features_fn(self):
        return self._features_fn

    @property
    def cost_fn(self):
        return self._cost_fn

    @property
    def cost_runtime(self):
        return self._cost_runtime

    def reset(self):
        super().reset()
        # Cost & feature funcs defined post initialization
        self._cost_fn, self._cost_runtime = self.build_cost_fn()
        self._features_fn = self.env.features_fn

    def build_cost_fn(self):
        """Return two types of cost functions.

        Example:
            >>> cost = cost_fn(state, action)
            >>> cost = cost_runtime(state, action, weights)

        """
        env_feats_list = self.env.features_list
        # Pre-defined & runtime costs
        cost_fn = weigh_funcs(env_feats_list, self._cost_weights)
        cost_runtime = weigh_funcs_runtime(env_feats_list)
        return cost_fn, cost_runtime

    def control(self, u, dt):
        assert (
            self._features_fn is not None
            and self._cost_fn is not None
            and self._cost_runtime is not None
        ), "Need to initialize car by `env.reset()`"

        diff = self.dynamics_fn(self._state, u)
        self._state += self.dynamics_fn(self._state, u) * dt

    def copy(self):
        cost_weights = deepcopy(list(self._cost_weights))
        return OptimalControlCar(
            self.env, cost_weights, deepcopy(self._init_state), self.horizon, self.color
        )


class UserControlCar(Car):
    def __init__(self, init_state, color="yellow", force=1.0):
        self._force = force
        super().__init__(init_state, horizon=1, color=color)
        pass
