import jax.numpy as jnp
import jax
import pyglet
import numpy as onp
import traceback, sys
from rdb.envs.drive2d.core.utils import centered_image
from rdb.envs.drive2d.core.dynamics import *
from rdb.optim.utils import *
from copy import deepcopy


def car_sprite(color, scale=0.15 / 600.0, batch=None, group=None):
    sprite = pyglet.sprite.Sprite(
        centered_image("car-{}.png".format(color)),
        subpixel=True,
        group=group,
        batch=batch,
    )
    sprite.scale = scale
    return sprite


def truck_sprite(scale=0.3 / 600.0, batch=None, group=None):
    sprite = pyglet.sprite.Sprite(
        centered_image("firetruck.png"), subpixel=True, group=group, batch=batch
    )
    sprite.scale = scale
    return sprite

def motorcycle_sprite(scale=0.2 / 600.0, batch=None, group=None):
    sprite = pyglet.sprite.Sprite(
        centered_image("motorcycle.png"), subpixel=True, group=group, batch=batch
    )
    sprite.scale = scale
    return sprite

class Car(object):
    def __init__(self, env, init_state, horizon, color, friction=0.1):
        """General Car Object.

        Args:
            init_state: initial state
            horizon: planning horizon

        Properties:
            state: (nbatch, udim), by default nbatch = 1

        """
        self.env = env
        self.horizon = horizon
        self.dynamics_fn = build_car_dynamics(friction)
        if len(init_state.shape) == 1:
            self._init_state = init_state[None, :]
            self._state = init_state[None, :]
        else:
            self._init_state = init_state
            self._state = init_state
        self._sprite = None
        self._color = color
        self._xdim = 4
        self._udim = 2
        self._curr_control = None
        # Control bound
        self._max_throttle = onp.inf
        self._max_brake = onp.inf
        self._max_steer = onp.inf
        # Speed bound
        self._max_speed = onp.inf
        # Initilialize trajectory data

    @property
    def xdim(self):
        return self._xdim

    @property
    def udim(self):
        return self._udim

    def reset(self):
        self.state = self._init_state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        assert len(state.shape) == 2
        self._state = state
        if self._sprite is not None:
            self._sprite.x, self._sprite.y = state[0, 0], state[0, 1]
            self._sprite.rotation = -state[0, 2] * 180 / onp.pi

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, init_state):
        assert len(init_state.shape) == 2
        self._init_state = init_state

    @property
    def color(self):
        return self._color

    @property
    def brake(self):
        if self._curr_control is not None:
            if self._state[0, 3] > 0:  # Forward
                return max(0, float(-1 * self._curr_control[0][1]))
            else:  # Backward
                return max(0, float(self._curr_control[0][1]))
        else:
            return 0

    @property
    def throttle(self):
        if self._curr_control is not None:
            if self._state[0, 3] > 0:  # Forward
                return max(0, float(self._curr_control[0][1]))
            else:  # Backward
                return max(0, float(-1 * self._curr_control[0][1]))
        else:
            return 0

    @property
    def steer(self):
        if self._curr_control is not None:
            return onp.abs(float(self._curr_control[0][0]))
        else:
            return 0

    @property
    def max_throttle(self):
        return self._max_throttle

    @max_throttle.setter
    def max_throttle(self, throttle):
        self._max_throttle = throttle

    @property
    def max_steer(self):
        return self._max_steer

    @max_steer.setter
    def max_steer(self, steer):
        self._max_steer = steer

    @property
    def max_brake(self):
        return self._max_brake

    @max_brake.setter
    def max_brake(self, brake):
        self._max_brake = brake

    @property
    def max_speed(self):
        return self._max_speed

    @max_speed.setter
    def max_speed(self, max_speed):
        self._max_speed = max_speed

    def control(self, u, dt):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def register(self, batch, group, opacity=255):
        """Register render layer"""
        self._sprite = car_sprite(self.color, batch=batch, group=group)
        self._sprite.opacity = opacity
        self._sprite.rotation = -self._state[0, 2] * 180 / onp.pi
        state = self._state
        self._sprite.x, self._sprite.y = state[0, 0], state[0, 1]


class FixSpeedCar(Car):
    def __init__(self, env, init_state, fix_speed, horizon=1, color="red"):
        self.env = env
        super().__init__(env, init_state, horizon, color)
        self.fix_speed = fix_speed
        self.dynamics_fn = build_fixspeed_dynamics(fix_speed)

    def control(self, dt):
        self.state = (
            self.state
            + self.dynamics_fn(self._state, jnp.zeros((len(self._state), self.udim)))
            * dt
        )
        self._curr_control = onp.zeros((1, 2))

    def copy(self):
        return FixSpeedCar(
            self.env,
            deepcopy(self._init_state),
            self.fix_speed,
            self.horizon,
            self.color,
        )


class FixSpeedTruck(FixSpeedCar):
    def __init__(self, env, init_state, fix_speed, horizon=1):
        super().__init__(env, init_state, fix_speed, horizon)
        pass

    def register(self, batch, group, opacity=255):
        """Register render layer"""
        self._sprite = truck_sprite(batch=batch, group=group)
        self._sprite.opacity = opacity
        self._sprite.rotation = -self._state[0, 2] * 180 / onp.pi
        state = self._state
        self._sprite.x, self._sprite.y = state[0, 0], state[0, 1]

class FixSpeedMotorcycle(FixSpeedCar):
    def __init__(self, env, init_state, fix_speed, horizon=1):
        super().__init__(env, init_state, fix_speed, horizon)
        pass

    def register(self, batch, group, opacity=255):
        """Register render layer"""
        self._sprite = motorcycle_sprite(batch=batch, group=group)
        self._sprite.opacity = opacity
        self._sprite.rotation = -self._state[0, 2] * 180 / onp.pi
        state = self._state
        self._sprite.x, self._sprite.y = state[0, 0], state[0, 1]


class OptimalControlCar(Car):
    def __init__(self, env, init_state, horizon=10, color="yellow"):
        """Autonomous car with optimal controller.

        Args:
            env: world

        """
        super().__init__(env, init_state, horizon, color)
        self._features_fn = None
        self._cost_runtime = None

    @property
    def features_fn(self):
        return self._features_fn

    @property
    def cost_runtime(self):
        return self._cost_runtime

    def reset(self):
        super().reset()
        # Cost & feature funcs defined post initialization
        self._cost_runtime = self.build_cost_fn()
        self._features_fn = self.env.features_fn

    def build_cost_fn(self):
        """Return two types of cost functions.

        Example:
            >>> cost = cost_fn(state, action)
            >>> cost = cost_runtime(state, action, weights)

        """
        env_feats_dict = self.env.features_dict
        # Runtime costs
        cost_runtime = weigh_funcs_runtime(env_feats_dict)
        return cost_runtime

    def control(self, u, dt):
        assert (
            self._features_fn is not None and self._cost_runtime is not None
        ), "Need to initialize car by `env.reset()`"

        self._curr_control = onp.array(u)
        diff = self.dynamics_fn(self._state, u)
        self.state = self.state + self.dynamics_fn(self._state, u) * dt

    def copy(self):
        return OptimalControlCar(
            self.env, deepcopy(self._init_state), self.horizon, self.color
        )


class UserControlCar(Car):
    def __init__(self, init_state, color="red", force=1.0):
        self._force = force
        super().__init__(init_state, horizon=1, color=color)
        pass
