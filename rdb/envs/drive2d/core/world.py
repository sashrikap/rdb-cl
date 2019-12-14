"""Generic driving world.

TODO:
    * All cars are assumed same size (width, length), and width/length are env features

"""


import abc
import gym
import jax
import jax.numpy as np
import numpy as onp
from pathlib import Path
from os.path import join
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
from numpyro.handlers import seed

import pyglet
import matplotlib.cm
from pyglet import gl, graphics
from gym.envs.classic_control import rendering
from rdb.envs.drive2d.core import lane, feature, car
from rdb.optim.utils import *

import platform

SYSTEM = platform.system()

WINDOW_W = 1200
WINDOW_H = 1200
pyglet.resource.path.append(str(Path(__file__).parent.parent.joinpath("assets")))


class DriveWorld(gym.Env):
    """General driving world.

    Attributes:
        main_car: the main autonomous car
        cars: a list of cars
        objects: a list of objects
        dt: timestep

    """

    def __init__(
        self,
        main_car,
        cars,
        lanes,
        dt,
        objects=[],
        subframes=3,
        car_length=0.1,
        car_width=0.075,
        speed_factor=100,
    ):
        self._main_car = main_car
        self._cars = cars
        self._lanes = lanes
        self._objects = objects
        self._dt = dt
        self._udim = 2
        self._car_length = car_length
        self._car_width = car_width
        self._indices = None
        self._speed_factor = speed_factor  # real world speed (mph) per simulation speed

        # Subframe rendering
        self._subframes = subframes
        self._sub_cars = [car.copy() for car in self._cars]
        self._sub_main_car = self.main_car.copy()
        self._prev_cars = [car.copy() for car in self._cars]
        self._prev_main_car = self.main_car.copy()

        # Render specifics
        self._window = None
        self._label = None
        self._grass = None
        self._magnify = 1.0
        self._cm = matplotlib.cm.coolwarm
        self._heat_fn = None

        self.xdim = onp.prod(self.state.shape)
        self._dynamics_fn, self._indices = self._get_dynamics_fn()
        self._raw_features_dict, self._raw_features_list, self._features_list, self._features_fn = (
            self._get_features_fn()
        )
        self._compile()
        self._constraints_dict, self._constraints_fn = self._get_constraints_fn()
        self._metadata_dict, self._metadata_fn = self._get_metadata_fn()

        # For sampling tasks
        self._rng_key = None
        self._all_tasks = None
        self._grid_tasks = None

    @property
    def subframes(self):
        return self._subframes

    @property
    def dt(self):
        return self._dt

    @property
    def udim(self):
        return self._udim

    @property
    def state(self):
        cars = self._cars + [self.main_car]
        state = []
        for car in cars:
            state.append(car.state)
        return np.concatenate(state)

    @state.setter
    def state(self, state):
        cars = self._cars + [self.main_car]
        last_idx = 0
        for car in cars:
            car.state = state[last_idx : last_idx + len(car.state)]
            last_idx += len(car.state)

    def set_task(self, state):
        raise NotImplementedError

    def set_init_state(self, state):
        """Set initial state."""
        cars = self._cars + [self.main_car]
        last_idx = 0
        for car in cars:
            car.init_state = state[last_idx : last_idx + len(car.state)]
            last_idx += len(car.state)
        self.state = state

    @property
    def features_list(self):
        """Nonlinear features dict."""
        return self._features_list

    @property
    def features_fn(self):
        """Nonlinear dictionary features function."""
        return self._features_fn

    @property
    def constraints_fn(self):
        return self._constraints_fn

    @property
    def constraints_dict(self):
        return self._constraints_dict

    @property
    def metadata_fn(self):
        return self._metadata_fn

    @property
    def raw_features_list(self):
        return self._raw_features_list

    @property
    def raw_features_dict(self):
        return self._raw_features_dict

    @property
    def dynamics_fn(self):
        return self._dynamics_fn

    @property
    def cars(self):
        return self._cars

    @property
    def main_car(self):
        return self._main_car

    @property
    def indices(self):
        return self._indices

    @property
    def car_width(self):
        return self._car_width

    @property
    def car_length(self):
        return self._car_length

    @property
    def features_keys(self):
        return ["dist_cars", "dist_lanes", "speed", "control"]

    @property
    def constraints_keys(self):
        return ["collision", "overspeed", "underspeed", "uncomfortable"]

    def _compile(self):
        self._dynamics_fn = jax.jit(self._dynamics_fn)
        self._features_fn = jax.jit(self._features_fn)

    def _get_dynamics_fn(self):
        """Build Dict(key: dynamics_fn) mapping.

        Example:
            >>> feat_1 = feature_fn_1(state)
            >>> dyn_fn[indices['cars_0']](state) =  next_car0_s

        Output:
            indices["cars0"] : func idx
            indices["cars1"] : func idx
            indices["main_car"] : func idx

        """
        dynamics_keys = [f"cars{i}" for i in range(len(self._cars))] + ["main_car"]
        fns, indices = OrderedDict(), OrderedDict()
        cars = self._cars + [self.main_car]
        curr_idx, next_idx = 0, 0
        for key, car in zip(dynamics_keys, cars):
            next_idx += np.prod(car.state.shape)
            idx = (curr_idx, next_idx)
            curr_idx = next_idx
            # dynamics funcion
            fn = index_func(car.dynamics_fn, idx)
            fns[key] = fn
            indices[key] = idx
        dynamics_fn = concat_funcs(fns.values())
        return dynamics_fn, indices

    def _get_raw_features_dict(self):
        """Build dict(key: feature_fn) mapping.

        Example:
            >>> feat_1 = feature_fn_1(state)

        Output:
            indices["dist_cars"]: `dist = func(state)`
            indices["dist_lanes"]:
            indices["speed"]:
            indices["control"]:

        Require:
            self._indices: e.g. {'car0': (0, 3)}

        Output:
            feat_fns: e.g. {'dist_cars': lambda state: return dist}

        """
        assert self._indices is not None, "Need to define state indices"
        feats_dict = OrderedDict()
        # Car feature functions
        car_fns = [None] * len(self._cars)
        for c_i, car in enumerate(self._cars):
            key = f"cars{c_i}"
            car_idx = self._indices[key]
            main_idx = self._indices["main_car"]
            # car_shape = np.array([self._car_width, self._car_length])

            def car_dist_fn(state, actions, car_idx=car_idx):
                car_pos = state[..., np.arange(*car_idx)]
                main_pos = state[..., np.arange(*main_idx)]
                return feature.diff_to(car_pos, main_pos)

            car_fns[c_i] = car_dist_fn

        # Lane feature functions
        lane_fns = [None] * len(self._lanes)
        for l_i, lane in enumerate(self._lanes):
            main_idx = self._indices["main_car"]

            def lane_dist_fn(state, actions, lane=lane):
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist_to_lane(lane.center, lane.normal, main_pos)

            lane_fns[l_i] = lane_dist_fn

        def speed_fn(state, actions):
            return feature.speed_size(state[..., np.arange(*main_idx)])

        def control_fn(state, actions):
            return feature.control_magnitude(actions)

        feats_dict["dist_cars"] = concat_funcs(car_fns, axis=0)
        feats_dict["dist_lanes"] = concat_funcs(lane_fns, axis=0)
        feats_dict["speed"] = speed_fn
        feats_dict["control"] = control_fn

        return feats_dict

    def _get_raw_features(self):
        feats_dict = self._get_raw_features_dict()
        feats_list = list(
            sort_dict_based_on_keys(feats_dict, self.features_keys).values()
        )
        return feats_dict, feats_list

    def _get_features_fn(self):
        raw_feats_dict, raw_feats_list = self._get_raw_features()
        nlr_feats_list = self._get_nonlinear_features_list(raw_feats_list)
        # Pre-compile individual feature functions, for speed up
        for idx, fn in enumerate(nlr_feats_list):
            nlr_feats_list[idx] = jax.jit(fn)

        # One-input-multi-output
        nlr_feats_dict = OrderedDict(zip(self.features_keys, nlr_feats_list))
        merged_feats_dict_fn = merge_dict_funcs(nlr_feats_dict)
        # merged_feats_list_fn = merge_list_funcs(nlr_feats_list)

        return raw_feats_dict, raw_feats_list, nlr_feats_list, merged_feats_dict_fn

    def _get_nonlinear_features_list(self, feats_list):
        raise NotImplementedError

    def _get_constraints_fn(self):
        raise NotImplementedError

    def _get_metadata_fn(self):
        return {}, lambda x: x

    def update_key(self, rng_key):
        self._rng_key = rng_key

    @property
    def all_tasks(self):
        return self._all_tasks

    @property
    def grid_tasks(self):
        """ For plotting purpose, meshgrid version of all_tasks"""
        return self._grid_tasks

    def reset(self):
        for car in self._cars:
            car.reset()
        self.main_car.reset()

    def step(self, action):
        for car, prev_car in zip(self._cars, self._prev_cars):
            prev_car.state = deepcopy(car.state)
        self._prev_main_car.state = deepcopy(self.main_car.state)

        for car in self._cars:
            car.control(self.dt)
        self.main_car.control(action, self.dt)
        rew = 0
        done = False
        return self.state, rew, done, {}

    def render(
        self,
        mode="rgb_array",
        cars=None,
        main_car=None,
        text=None,
        draw_heat=False,
        weights=None,
    ):
        assert mode in ["human", "state_pixels", "rgb_array"]

        if self._window is None:
            caption = f"{self.__class__}"
            # JH Note: strange pyglet rendering requires /2
            self._window = pyglet.window.Window(
                width=int(WINDOW_W / 2), height=int(WINDOW_H / 2)
            )
        self._window.switch_to()
        # Bring up window
        self._window.dispatch_events()
        self._window.clear()
        if SYSTEM == "Darwin":
            gl.glViewport(0, 0, int(WINDOW_W), int(WINDOW_H))
        else:  # Linux, etc
            gl.glViewport(0, 0, int(WINDOW_W / 2), int(WINDOW_H / 2))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        if cars is None:
            cars = self._cars
        if main_car is None:
            main_car = self.main_car

        self.center_camera(main_car)
        self.draw_background()
        for lane in self._lanes:
            lane.render()
        for obj in self._objects:
            obj.render()
        for car in cars:
            car.render()
        main_car.render()
        if draw_heat:
            self.set_heat(weights)
            self.draw_heatmap()

        gl.glPopMatrix()
        self.draw_text(text)

        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = onp.fromstring(img_data.data, dtype=onp.uint8, sep="")
        if SYSTEM == "Darwin":
            arr = arr.reshape(int(WINDOW_W), int(WINDOW_H), 4)
        else:
            arr = arr.reshape(int(WINDOW_W / 2), int(WINDOW_H / 2), 4)
        arr = arr[::-1, :, 0:3]
        # if mode == "human":
        self._window.flip()
        return arr

    def sub_render(self, mode="rgb_array", subframe=0, text=None):
        """ Alpha-interpolate adjacent frames to make animation look smoother """
        ratio = (subframe + 1.0) / float(self._subframes)
        for c_i in range(len(self._cars)):
            diff_state = self._cars[c_i].state - self._prev_cars[c_i].state
            self._sub_cars[c_i].state = ratio * diff_state + self._prev_cars[c_i].state

        diff_state = self._main_car.state - self._prev_main_car.state
        self._sub_main_car.state = ratio * diff_state + self._prev_main_car.state
        return self.render(
            mode=mode, cars=self._sub_cars, main_car=self._sub_main_car, text=text
        )

    def center_camera(self, main_car):
        center_x = main_car.state[0]
        center_y = main_car.state[1]
        gl.glOrtho(
            center_x - 1.0 / self._magnify,
            center_x + 1.0 / self._magnify,
            center_y - 1.0 / self._magnify,
            center_y + 1.0 / self._magnify,
            -1.0,
            1.0,
        )

    def draw_background(self):
        if self._grass is None:
            self._grass = pyglet.resource.texture("grass.png")
        gl.glEnable(self._grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self._grass.target, self._grass.id)
        W = 10000.0
        graphics.draw(
            4,
            gl.GL_QUADS,
            ("v2f", (-W, -W, W, -W, W, W, -W, W)),
            ("t2f", (0.0, 0.0, W * 5.0, 0.0, W * 5.0, W * 5.0, 0.0, W * 5.0)),
        )
        gl.glDisable(self._grass.target)

    def draw_text(self, text=None):
        if self._label is None:
            assert self._window is not None
            self._label = pyglet.text.Label(
                "Speed: ",
                font_name="Palatino",
                font_size=15,
                x=30,
                y=self._window.height - 30,
                width=200,
                anchor_x="left",
                anchor_y="top",
                multiline=True,
            )
        speed = self._main_car.state[3] * self._speed_factor
        self._label.text = f"Speed: {speed:.2f} mph"
        if text is not None:
            self._label.text += "\n" + text
        self._label.draw()

    def draw_lane(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        normal, forward = lane.normal, lane.forward
        pt1, pt2, width = lane.pt1, lane.pt2, lane.width
        quad_strip = onp.hstack(
            [
                pt1 - forward * W - 0.5 * width * normal,
                pt1 - forward * W + 0.5 * width * normal,
                pt2 + forward * W - 0.5 * width * normal,
                pt2 + forward * W + 0.5 * width * normal,
            ]
        )
        graphics.draw(4, gl.GL_QUAD_STRIP, ("v2f", quad_strip))
        gl.glColor3f(1.0, 1.0, 1.0)
        W = 1000
        line_strip = onp.hstack(
            [
                pt1 - forward * W - 0.5 * width * normal,
                pt1 + forward * W - 0.5 * width * normal,
                pt1 - forward * W + 0.5 * width * normal,
                pt1 + forward * W + 0.5 * width * normal,
            ]
        )
        graphics.draw(4, gl.GL_LINES, ("v2f", line_strip))

    def set_heat(self, weights=None):
        def val(x, y):
            if weights is None:
                cost_fn = self._main_car.cost_fn
            else:
                cost_fn = partial(self._main_car.cost_runtime, weights=weights)
            state = deepcopy(self.state)
            main_idx = self._indices["main_car"]
            state[main_idx[0] : main_idx[0] + 3] = [x, y, onp.pi / 3]
            act = onp.array([0, 0])
            return cost_fn(state, act)

        self._heat_fn = val

    def draw_heatmap(self):
        center = self.main_car.state[:2]
        c0 = center - onp.array([1.0, 1.0]) / self._magnify
        c1 = center + onp.array([1.0, 1.0]) / self._magnify

        SIZE = (32, 32)
        # SIZE = (16, 16)
        vals = onp.zeros(SIZE)
        # Sweep for cost values
        for i, x in enumerate(tqdm(onp.linspace(c0[0], c1[0], SIZE[0]))):
            for j, y in enumerate(onp.linspace(c0[1], c1[1], SIZE[1])):
                vals[j, i] = self._heat_fn(x, y)
        vals = (vals - onp.min(vals)) / (onp.max(vals) - onp.min(vals) + 1e-6)
        # Convert to color map and draw
        vals = self._cm(vals)
        vals[:, :, 3] = 0.5
        vals = (vals * 255.99).astype("uint8").flatten()
        vals = (gl.GLubyte * vals.size)(*vals)
        img = pyglet.image.ImageData(SIZE[0], SIZE[1], "RGBA", vals, pitch=SIZE[1] * 4)
        heatmap = img.get_texture()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(heatmap.target, heatmap.id)
        graphics.draw(
            4,
            gl.GL_QUADS,
            ("v2f", (c0[0], c0[1], c1[0], c0[1], c1[0], c1[1], c0[0], c1[1])),
            ("t2f", (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
        )
        gl.glDisable(heatmap.target)


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


def object_sprite(name, scale=0.15 / 600.0):
    sprite = pyglet.sprite.Sprite(centered_image("{}.png".format(name)), subpixel=True)
    sprite.scale = scale
    return sprite
