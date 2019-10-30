import jax.numpy as np
from rdb.envs.drive2d.core.dynamics import *
from rdb.optim.utils import *
from copy import deepcopy


class Car(object):
    def __init__(self, env, init_state, horizon, color, friction=0.1):
        """
        Params
        : init_state : initial state
        : horizon    : planning horizon
        """
        self.env = env
        self.horizon = horizon
        self.dynamics_fn = car_dynamics_fn(friction)
        self.init_state = init_state
        self.state = init_state
        self.color = color
        # Initilialize trajectory data

    def reset(self):
        self.state = self.init_state

    def control(self, u, dt):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class FixSpeedCar(Car):
    def __init__(self, env, init_state, fix_speed, horizon=1, color="red"):
        self.env = env
        super().__init__(env, init_state, horizon, color)
        self.fix_speed = fix_speed
        self.dynamics_fn = fixspeed_dynamics_fn(fix_speed)

    def control(self, dt):
        self.state += self.dynamics_fn(self.state, None) * dt

    def copy(self):
        return FixSpeedCar(
            self.env,
            deepcopy(self.init_state),
            self.fix_speed,
            self.horizon,
            self.color,
        )


class OptimalControlCar(Car):
    def __init__(self, env, cost_weights, init_state, horizon=10, color="yellow"):
        """
        Params
        : env : world
        """
        super().__init__(env, init_state, horizon, color)
        self.cost_weights = cost_weights
        self.features = None
        self.cost_fn = None

    def reset(self):
        super().reset()
        self.cost_fn = self.build_cost_fn(self.env.cost_features, self.cost_weights)

    def build_cost_fn(self, feature_fns, weights):
        """ cost = cost_fn(state, action)
        """
        env_feat_fns = self.env.feat_fns
        cost_feat_fns = chain_funcs(feature_fns, env_feat_fns)
        cost_fn = weigh_funcs(cost_feat_fns, weights)
        return cost_fn

    def control(self, u, dt):
        diff = self.dynamics_fn(self.state, u)
        self.state += self.dynamics_fn(self.state, u) * dt

    def copy(self):
        return OptimalControlCar(
            self.env,
            deepcopy(self.cost_weights),
            deepcopy(self.init_state),
            self.horizon,
            self.color,
        )


class UserControlCar(Car):
    def __init__(self, init_state, color="yellow", force=1.0):
        self._force = force
        super().__init__(init_state, horizon=1, color=color)
        pass
