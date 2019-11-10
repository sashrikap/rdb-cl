import abc
import gym
import jax.numpy as np
import numpy as onp
from pathlib import Path
from os.path import join
import copy
from collections import OrderedDict

import pyglet
from pyglet import gl, graphics
from gym.envs.classic_control import rendering
from rdb.envs.drive2d.core import lane, feature, car
from rdb.optim.utils import *


WINDOW_W = 1200
WINDOW_H = 1200
pyglet.resource.path.append(str(Path(__file__).parent.parent.joinpath("assets")))


class DriveWorld(gym.Env):
    """
    General driving world

    Key Attributes
    : main_car :
    : cars     :
    : objects  :
    : dt       : timestep
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
        car_width=0.08,
    ):
        self._main_car = main_car
        self._cars = cars
        self._lanes = lanes
        self._objects = objects
        self._dt = dt
        self._car_sprites = {c.color: car_sprite(c.color) for c in cars + [main_car]}
        self._obj_sprites = {o.name: object_sprite(o.name) for o in objects}
        self._car_length = car_length
        self._car_width = car_width
        self._indices = None

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

        self.xdim = np.prod(self.state.shape)
        self._dynamics_fn, self._indices = self.get_dynamics_fn()
        self._features_dict, self._features_fn = self.get_features_fn()

    @property
    def subframes(self):
        return self._subframes

    @property
    def dt(self):
        return self._dt

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

    @property
    def features_dict(self):
        return self._features_dict

    @property
    def features_fn(self):
        return self._features_fn

    @property
    def dynamics_fn(self):
        return self._dynamics_fn

    @property
    def cars(self):
        return self._cars

    @property
    def main_car(self):
        return self._main_car

    def get_dynamics_fn(self):
        """ Build Dict(key: dynamics_fn) mapping
        Usage
        ``` feat_1 = feature_fn_1(state) ```
        Keys:
        : cars0    :
        : cars1    :
        : main_car :
        Output
        : dynamic_fns : e.g. dyn_fns['cars_0'](state) =  next_car0_s
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

    def get_raw_features_dict(self):
        """ Build Dict(key: feature_fn) mapping
        Usage
            ``` feat_1 = feature_fn_1(state) ```
        Keys:
        : dist_cars  :
        : dist_lanes :
        : speed      :
        : control    :
        Param
        : indices  : e.g. {'car0': (0, 3)}
        Output
        : feat_fns : e.g. {'dist_cars': lambda state: return dist}
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

        feats_dict["dist_cars"] = concat_funcs(car_fns)
        feats_dict["dist_lanes"] = concat_funcs(lane_fns)
        feats_dict["speed"] = speed_fn
        feats_dict["control"] = control_fn
        return feats_dict

    def get_features_fn(self):
        feats_dict = self.get_raw_features_dict()
        feats_dict = self.get_nonlinear_features_dict(feats_dict)
        merged_fn = merge_dict_funcs(feats_dict)
        return feats_dict, merged_fn

    def get_nonlinear_features_dict(self, feats_dict):
        raise NotImplementedError

    def reset(self):
        for car in self._cars:
            car.reset()
        self.main_car.reset()

    def step(self, action):
        for car, prev_car in zip(self._cars, self._prev_cars):
            prev_car.state = copy.deepcopy(car.state)
        self._prev_main_car.state = copy.deepcopy(self.main_car.state)

        for car in self._cars:
            car.control(self.dt)
        self.main_car.control(action, self.dt)
        rew = 0
        done = False
        return self.state, rew, done, {}

    def render(self, mode="rgb_array", cars=None, main_car=None):
        assert mode in ["human", "state_pixels", "rgb_array"]

        if self._window is None:
            caption = f"{self.__class__}"
            # JH Note: strange pyglet rendering requires /2
            self._window = pyglet.window.Window(
                width=int(WINDOW_W / 2),
                height=int(WINDOW_H / 2)
                # width=int(WINDOW_W), height=int(WINDOW_H),
            )
        self._window.switch_to()
        # if mode == "human":
        # Bring up window
        self._window.dispatch_events()
        self._window.clear()
        gl.glViewport(0, 0, int(WINDOW_W), int(WINDOW_H))
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
            self.draw_lane(lane)
        for obj in self._objects:
            self.draw_object(obj)
        for car in cars:
            self.draw_car(car)
        self.draw_car(main_car)
        gl.glPopMatrix()
        self.draw_text()

        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = onp.fromstring(img_data.data, dtype=onp.uint8, sep="")
        # arr = arr.reshape(int(WINDOW_W / 2), int(WINDOW_H / 2), 4)
        arr = arr.reshape(int(WINDOW_W), int(WINDOW_H), 4)
        arr = arr[::-1, :, 0:3]
        # if mode == "human":
        self._window.flip()
        return arr

    def sub_render(self, mode="rgb_array", subframe=0):
        """ Alpha-interpolate adjacent frames to make animation look smoother """
        ratio = (subframe + 1.0) / float(self._subframes)
        for c_i in range(len(self._cars)):
            diff_state = self._cars[c_i].state - self._prev_cars[c_i].state
            self._sub_cars[c_i].state = ratio * diff_state + self._prev_cars[c_i].state

        diff_state = self._main_car.state - self._prev_main_car.state
        self._sub_main_car.state = ratio * diff_state + self._prev_main_car.state
        return self.render(mode=mode, cars=self._sub_cars, main_car=self._sub_main_car)

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

    def draw_object(self, obj, opacity=255):
        sprite = self._obj_sprites[obj.name]
        sprite.x, sprite.y = obj.state[1], obj.state[0]
        rotation = 0.0
        if len(obj.state) >= 3:
            rotation = obj.state[2]
        sprite.rotation = rotation
        sprite.opacity = opacity
        sprite.draw()

    def draw_car(self, car, opacity=255):
        sprite = self._car_sprites[car.color]
        sprite.x, sprite.y = car.state[0], car.state[1]
        # sprite.x, sprite.y = 0, 0
        sprite.rotation = -car.state[2] * 180 / onp.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_text(self):
        if self._label is None:
            assert self._window is not None
            self._label = pyglet.text.Label(
                "Speed: ",
                font_name="Times New Roman",
                font_size=24,
                x=30,
                y=self._window.height - 30,
                anchor_x="left",
                anchor_y="top",
            )
        self._label.text = f"Speed: {self._main_car.state[3]:.3f}"
        self._label.draw()

    def draw_lane(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000
        normal, forward = lane.normal, lane.forward
        pt1, pt2, width = lane.pt1, lane.pt2, lane.width
        quad_strip = np.hstack(
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
        line_strip = np.hstack(
            [
                pt1 - forward * W - 0.5 * width * normal,
                pt1 + forward * W - 0.5 * width * normal,
                pt1 - forward * W + 0.5 * width * normal,
                pt1 + forward * W + 0.5 * width * normal,
            ]
        )
        graphics.draw(4, gl.GL_LINES, ("v2f", line_strip))


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
