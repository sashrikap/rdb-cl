import abc
import gym
import jax.numpy as np
import numpy as onp
from pathlib import Path
from os.path import join
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
    def __init__(self, main_car, cars, lanes, dt, objects=[]):
        self._main_car = main_car
        self._cars = cars
        self._lanes = lanes
        self._objects = objects
        self._dt = dt
        self._car_sprites = {c.color: car_sprite(c.color) for c in cars + [main_car]}
        self._obj_sprites = {o.name: object_sprite(o.name) for o in objects}
        self.subframes = 6

        # Render specifics
        self._window = None
        self._label = None
        self._grass = None
        self._magnify = 1.0

        self.xdim = np.prod(self.state.shape)
        dynamics_fns, indices = self.get_dynamics_fns()
        self.dynamics_fn = concat_funcs(dynamics_fns.values())
        self.indices = indices
        feat_fns = self.get_feat_fns(indices)
        self.feat_fns = feat_fns

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

    def get_dynamics_fns(self):
        """ Build Dict(key: dynamics_fn) mapping

        Keys: cars0, cars1, main_car
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
        return fns, indices

    def get_feat_fns(self, indices):
        """ Build Dict(key: feature_fn) mapping

        Param:
        : masks : state masks
        """
        fns = OrderedDict()
        # Car feature functions
        car_fns = [None] * len(self._cars)
        for c_i, car in enumerate(self._cars):
            key = f"cars{c_i}"
            car_idx = indices[key]
            main_idx = indices["main_car"]

            def car_dist_fn(state, actions, car_idx=car_idx):
                car_pos = state[..., np.arange(*car_idx)]
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist2(car_pos, main_pos)

            car_fns[c_i] = car_dist_fn

        # Lane feature functions
        lane_fns = [None] * len(self._lanes)
        for l_i, lane in enumerate(self._lanes):
            main_idx = indices["main_car"]

            def lane_dist_fn(state, actions, lane=lane):
                main_pos = state[..., np.arange(*main_idx)]
                return feature.dist2lane(lane.center, lane.normal, main_pos)

            lane_fns[l_i] = lane_dist_fn

        def speed_fn(state, actions):
            return feature.speed_size(state[..., np.arange(*main_idx)])

        def control_fn(state, actions):
            return feature.control_magnitude(actions)

        fns["dist_cars"] = concat_funcs(car_fns)
        fns["dist_lanes"] = concat_funcs(lane_fns)
        fns["speed"] = speed_fn
        fns["control"] = control_fn
        return fns

    @property
    def cars(self):
        return self._cars

    @property
    def main_car(self):
        return self._main_car

    def reset(self):
        for car in self._cars:
            car.reset()
        self.main_car.reset()

    def step(self, action):
        for car in self._cars:
            car.control(self.dt)
        self.main_car.control(action, self.dt)
        rew = 0
        done = False
        return self.state, rew, done, {}

    def render(self, mode="rgb_array"):
        assert mode in ["human", "state_pixels", "rgb_array"]

        if self._window is None:
            caption = f"{self.__class__}"
            # JH Note: strange pyglet rendering requires /2
            self._window = pyglet.window.Window(
                width=int(WINDOW_W / 2), height=int(WINDOW_H / 2)
            )
        self._window.switch_to()
        if mode == "human":
            # Bring up window
            self._window.dispatch_events()
        self._window.clear()
        # gl.glViewport(0, 0, int(WINDOW_W / 2), int(WINDOW_H / 2))
        gl.glViewport(0, 0, int(WINDOW_W), int(WINDOW_H))
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        self.draw_background()
        for lane in self._lanes:
            self.draw_lane(lane)
        for obj in self._objects:
            self.draw_object(obj)
        for car in self._cars:
            self.draw_car(car)
        self.draw_car(self.main_car)
        gl.glPopMatrix()

        img_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = onp.fromstring(img_data.data, dtype=onp.uint8, sep="")
        # arr = arr.reshape(int(WINDOW_W / 2), int(WINDOW_H / 2), 4)
        arr = arr.reshape(int(WINDOW_W), int(WINDOW_H), 4)
        arr = arr[::-1, :, 0:3]
        if mode == "human":
            self._window.flip()
        return arr

    def center_camera(self):
        center_x = self.main_car.state[0]
        center_y = self.main_car.state[1]
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
        self.center_camera()
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
        self.label.text = "Speed: 0"
        self.label.draw()

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
