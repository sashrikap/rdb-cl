"""Generic driving world.

TODO:
    * All cars are assumed same size (width, length), and width/length are env features

"""


import abc
import gym
import jax
import jax.numpy as jnp
import numpy as onp
from pathlib import Path
from os.path import join
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm

# from numpyro.handlers import seed
from gym import spaces
import pyglet
from pyglet import gl, graphics

from rdb.envs.drive2d.core import lane, feature, car, utils
from rdb.visualize.render import RenderEnv
from rdb.infer.dictlist import *
from rdb.optim.utils import *
from rdb.exps.utils import Profiler

import platform

SYSTEM = platform.system()

WINDOW_W = 1200
WINDOW_H = 1200
pyglet.resource.path.append(str(Path(__file__).parent.parent.joinpath("assets")))


class DriveWorld(RenderEnv):
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
        car_length=0.16,
        car_width=0.075,
        speed_factor=100,
    ):
        RenderEnv.__init__(self, subframes=subframes)
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

        # Render specifics
        self._window = None
        self._headless = False
        self._grass = None
        self._magnify = 1.0
        self._cm = None
        self._viz_heat_fn = None
        self._viz_constraint_fn = None
        self._layers = None
        self._texts = None

        # Dyanmics, features, constraints and metadata functions
        self._xdim = onp.prod(self.state.shape)
        self._build_dynamics_fn()
        self._build_features_fn()
        # Compute violations, metadata and task naturalness
        self._build_constraints_fn()
        self._build_metadata_fn()
        self._compile()

        # For sampling tasks
        self._rng_key = None
        self._all_tasks = None
        self._all_task_difficulties = None
        self._grid_tasks = []

        ## Set up rendering
        self._labels = {}

        acs_high = onp.array([self._main_car.max_steer, self._main_car.max_throttle])
        acs_low = -1 * acs_high
        obs_high = onp.array([onp.inf] * self._xdim)
        self.action_space = spaces.Box(acs_low, acs_high, dtype=onp.float32)
        self.observation_space = spaces.Box(-1 * obs_high, obs_high, dtype=onp.float32)

    @property
    def dt(self):
        return self._dt

    @property
    def udim(self):
        return self._udim

    @property
    def xdim(self):
        return self._xdim

    @property
    def state(self):
        """Current state

        Output:
            out (ndarray): (1, xdim)

        """
        cars = self._cars + [self.main_car]
        state = []
        for car in cars:
            state.append(car.state)
        for obj in self._objects:
            state.append(obj.state)
        state = jnp.concatenate(state, axis=1)
        return state

    @state.setter
    def state(self, state):
        cars = self._cars + [self.main_car]
        last_idx = 0
        for car in cars:
            car.state = state[:, last_idx : last_idx + car.xdim]
            last_idx += car.xdim
        for obj in self._objects:
            obj.state = state[:, last_idx : last_idx + obj.xdim]
            last_idx += obj.xdim

    def set_task(self, state):
        raise NotImplementedError

    def set_init_state(self, state):
        """Set initial state."""
        if len(state.shape) == 1:
            state = state[None, :]
        cars = self._cars + [self._main_car]
        last_idx = 0
        for car in cars:
            car.init_state = state[:, last_idx : last_idx + car.xdim]
            last_idx += car.xdim
        for obj in self._objects:
            obj.state = state[:, last_idx : last_idx + obj.xdim]
            last_idx += obj.xdim
        # self.state = state

    def get_init_state(self, task):
        raise NotImplementedError

    def get_init_states(self, tasks):
        raise NotImplementedError

    @property
    def raw_features_fn(self):
        """Nonlinear dictionary features function."""
        return self._raw_features_fn

    @property
    def features_fn(self):
        """Nonlinear dictionary features function.

        Args:
            state
            action
        Output:
            dict (str): feature vector

        """
        return self._features_fn

    @property
    def max_feats_dict(self):
        return self._max_feats_dict

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
    def objects(self):
        return self._objects

    @property
    def raw_features_dict(self):
        return self._raw_features_dict

    @property
    def features_dict(self):
        return self._features_dict

    @property
    def raw_features_keys(self):
        return tuple(self._raw_features_dict.keys())

    @property
    def features_keys(self):
        """Nonliear features keys"""
        return tuple(self._features_dict.keys())

    @property
    def constraints_keys(self):
        return tuple(self._constraints_dict.keys())

    @property
    def metadata_keys(self):
        return tuple(self._metadata_dict.keys())

    def _compile(self):
        self._dynamics_fn = jax.jit(self._dynamics_fn)
        self._features_fn = jax.jit(self._features_fn)
        self._constraints_fn = jax.jit(self._constraints_fn)
        self._metadata_fn = jax.jit(self._metadata_fn)

    def _build_dynamics_fn(self):
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
            next_idx += onp.prod(car.state.shape)
            idx = (curr_idx, next_idx)
            curr_idx = next_idx
            # dynamics funcion
            fn = index_func(car.dynamics_fn, idx)
            fns[key] = fn
            indices[key] = idx
        for o_i, obj in enumerate(self._objects):
            next_idx += onp.prod(obj.state.shape)
            idx = (curr_idx, next_idx)
            curr_idx = next_idx
            key = f"{obj.name}_{o_i:02d}"
            fn = index_func(obj.dynamics_fn, idx)
            fns[key] = fn
            indices[key] = idx
        self._dynamics_fn = concat_funcs(fns.values(), axis=1)
        self._indices = indices

    def _get_raw_features_dict(self):
        """Build dict(key: feature_fn) mapping.

        Note:
            Decides environment feature key order for raw_feats, nonlinear_feats.

        Output:
            fn["dist_cars"]: difference to other cars
                (nbatch, xdim) -> (nbatch, ncars, car_xdim)
            fn["dist_lanes"]: distance to lanes
                (nbatch, xdim) -> (nbatch, nlanes)
            fn["dist_objects"]: difference to objects
                (nbatch, xdim) -> (nbatch, nobjs, obj_xdim)
            fn["speed"]: speed magnitude
                (nbatch, xdim) -> (nbatch, 1)
            fn["control"]: control magnitude
                (nbatch, xdim) -> (nbatch, 1)

        Note:
            * requires self._indices: e.g. {'car0': (0, 3)}


        """
        assert self._indices is not None, "Need to define state indices"
        feats_dict = OrderedDict()
        # Car feature functions
        car_fns = [None] * len(self._cars)
        main_idx = self._indices["main_car"]
        for c_i, car in enumerate(self._cars):
            key = f"cars{c_i}"
            car_idx = self._indices[key]

            def car_dist_fn(state, actions, car_idx=car_idx):
                # Very important to keep third argument for variable closure
                car_state = state[..., jnp.arange(*car_idx)]
                main_state = state[..., jnp.arange(*main_idx)]
                return feature.diff_to(car_state, main_state)

            car_fns[c_i] = car_dist_fn

        # Lane feature functions
        lane_fns = [None] * len(self._lanes)
        for l_i, lane in enumerate(self._lanes):

            def lane_dist_fn(state, actions, lane=lane):
                main_state = state[..., jnp.arange(*main_idx)]
                dist = feature.dist_to_lane(main_state, lane.center, lane.normal)
                return dist

            lane_fns[l_i] = lane_dist_fn

        # Object feature functions
        obj_fns = [None] * len(self._objects)
        for o_i, obj in enumerate(self._objects):
            obj_idx = self._indices[f"{obj.name}_{o_i:02d}"]

            def obj_dist_fn(state, actions, obj_idx=obj_idx):
                main_state = state[..., jnp.arange(*main_idx)]
                obj_state = state[..., jnp.arange(*obj_idx)]
                return feature.diff_to(main_state, obj_state)

            obj_fns[o_i] = obj_dist_fn

        # Speed feature function
        def speed_fn(state, actions):
            return feature.speed_size(state[..., jnp.arange(*main_idx)])

        # Control feature function
        def control_mag_fn(state, actions):
            return feature.control_magnitude(actions)

        def control_throttle_fn(state, actions):
            return feature.control_throttle(actions)

        def control_brake_fn(state, actions):
            return feature.control_brake(actions)

        def control_turn_fn(state, actions):
            return feature.control_turn(actions)

        def bias_fn(state, actions):
            return feature.ones(state[..., jnp.arange(*main_idx)])

        # TODO add if checks for this, or some param that prevents having to comment out explicitly
        feats_dict["dist_cars"] = stack_funcs(car_fns, axis=1)
        feats_dict["dist_lanes"] = stack_funcs(lane_fns, axis=1)
        feats_dict["dist_objects"] = stack_funcs(obj_fns, axis=1)
        feats_dict["dist_obstacles"] = stack_funcs(obj_fns, axis=1)
        # feats_dict["dist_trees"] = stack_funcs(obj_fns, axis=1)
        feats_dict["speed"] = speed_fn
        feats_dict["speed_over"] = speed_fn
        feats_dict["speed_under"] = speed_fn
        feats_dict["control"] = control_mag_fn
        feats_dict["control_throttle"] = control_throttle_fn
        feats_dict["control_brake"] = control_brake_fn
        feats_dict["control_turn"] = control_turn_fn
        feats_dict["bias"] = bias_fn

        return feats_dict

    def _build_features_fn(self):
        """Compute raw features and non-linear features.

        """
        raw_feats_dict = self._get_raw_features_dict()
        nlr_feats_dict, max_feats_dict = self._get_nonlinear_features_dict(
            raw_feats_dict
        )
        # nlr_feats_dict = sort_dict_by_keys(nlr_feats_dict, raw_feats_dict.keys())
        # Pre-compile individual feature functions, for speed up
        for key, fn in nlr_feats_dict.items():
            nlr_feats_dict[key] = jax.jit(fn)

        # One-input-multi-output
        self._raw_features_dict = raw_feats_dict
        self._raw_features_fn = merge_dict_funcs(raw_feats_dict)
        self._features_fn = merge_dict_funcs(nlr_feats_dict)
        self._features_dict = nlr_feats_dict
        # self._max_feats_dict = sort_dict_by_keys(max_feats_dict, raw_feats_dict.keys())
        self._max_feats_dict = max_feats_dict

    def _get_nonlinear_features_dict(self, feats_dict):
        raise NotImplementedError

    def _build_constraints_fn(self):
        return {}, lambda x: x

    def _build_metadata_fn(self):
        return {}, lambda x: x

    def _get_natural_tasks(self, tasks):
        return tasks

    def update_key(self, rng_key):
        self._rng_key = rng_key

    def _setup_tasks(self):
        raise NotImplementedError

    @property
    def all_tasks(self):
        if self._all_tasks is None:
            self._setup_tasks()
        return self._all_tasks

    @property
    def all_task_difficulties(self):
        return self._all_task_difficulties

    @property
    def grid_tasks(self):
        """ For plotting purpose, meshgrid version of all_tasks"""
        return self._grid_tasks

    def reset(self):
        # super().reset()
        for car in self._cars:
            car.reset()
        self.main_car.reset()

    def step(self, action):
        use_batch = True
        if len(action.shape) == 1:
            action = onp.array([action])
            use_batch = False
        for car in self._cars:
            car.control(self.dt)
        self.main_car.control(action, self.dt)
        # compute all raw feats
        # compute all nlr feats
        # multiply nlr feats with weights -> rew
        # TODO: (Jerry) add cost computation back in
        rew = 0
        done = False
        obs = self.state
        if not use_batch:
            obs = obs[0]
        truncated = None  # compatibility for gym after python>=3.9
        return obs, rew, done, truncated, {}

    #####################################################################
    ##################### Rendering Functionalities #####################
    #####################################################################

    def render(
        self,
        mode="rgb_array",
        text=None,
        draw_heat=False,
        draw_boundary=False,
        draw_constraint_key=None,
        weights=None,
        paper=False,
    ):
        """
        Args:
            draw_heat (bool): draw heatmap, where redness ~ | cost |
            draw_boundary (bool): draw boundary map, where redness ~ cost > 0
        """
        assert mode in ["human", "rgb_array"]
        if self.state.shape[0] > 1:
            print("WARNING: rendering with batch mode")

        if paper:
            self._magnify = 1

        visible = mode == "human"
        ## Initialize rendering
        if self._window is None:
            # try:
            self._window = pyglet.window.Window(
                width=int(WINDOW_W / 2), height=int(WINDOW_H / 2), visible=visible
            )
            self._window.set_visible(True)
            self._setup_render(paper=paper)
            self._window.set_visible(visible)
            # except:
            #     ## On headless server
            #     print("Display not supported on headless server")
            #     self._headless = True
        if not self._headless:
            self._window.switch_to()
            # Bring up window
            self._window.dispatch_events()
            self._window.clear()
            if SYSTEM == "Darwin":
                # JH Note: strange pyglet rendering requires /2
                gl.glViewport(0, 0, int(WINDOW_W), int(WINDOW_H))
            else:  # Linux, etc
                gl.glViewport(0, 0, int(WINDOW_W / 2), int(WINDOW_H / 2))
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glPushMatrix()
            gl.glLoadIdentity()

            self._center_camera(self._main_car)
            self._layers.batch.draw()

            if draw_heat:
                assert weights is not None
                self._set_viz_heat(weights)
                self._draw_heatmap(heat_fn=self._viz_heat_fn)
            elif draw_boundary:
                assert weights is not None
                self._set_viz_heat(weights)
                self._draw_heatmap(
                    heat_fn=lambda *args: (self._viz_heat_fn(*args) > 0.01).astype(
                        float
                    )
                )
            elif draw_constraint_key is not None:
                self._set_viz_constraints(draw_constraint_key)
                self._draw_heatmap(heat_fn=self._viz_constraint_fn)

            gl.glPopMatrix()
            if not paper:
                self._update_text(text=text)
            # self._texts.batch.draw()

            img_data = (
                pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            )
            arr = onp.frombuffer(img_data.get_data(), dtype=onp.uint8)
            if SYSTEM == "Darwin":
                arr = arr.reshape(int(WINDOW_W), int(WINDOW_H), 4)
            else:
                arr = arr.reshape(int(WINDOW_W / 2), int(WINDOW_H / 2), 4)
            arr = arr[::-1, :, 0:3]
            # if mode == "human":
            self._window.flip()
            # self._window.set_visible(False)
            return arr
        else:
            return None

    def close_window(self):
        if self._window is not None:
            self._window.close()
        self._window = None

    def _center_camera(self, main_car):
        center_x = main_car.state[0, 0]
        center_y = main_car.state[0, 1]
        gl.glOrtho(
            center_x - 1.0 / self._magnify,
            center_x + 1.0 / self._magnify,
            center_y - 1.0 / self._magnify,
            center_y + 1.0 / self._magnify,
            -1.0,
            1.0,
        )

    def _setup_render(self, paper=False):
        ## Setup render batch
        self._layers = utils.EnvGroups()
        self._texts = utils.TextGroups()

        ## Setup background
        self._grass = pyglet.resource.texture("grass.png")
        W = 10

        ## Setup lanes
        # if paper:
        #     tex_verts = ("v2f", (-W, -W, W, -W, W, W, -W, W))
        #     tex_colors = ("c3B", [int(1 * 255)] * 12)
        #     self._layers.batch.add(4, gl.GL_QUADS, self._layers.background, tex_verts, tex_colors)
        # else:
        texture_grid = pyglet.image.ImageGrid(self._grass, W, W).get_texture_sequence()
        texture_group = pyglet.graphics.TextureGroup(
            texture_grid, self._layers.background
        )
        tex_map = texture_grid.get_texture().tex_coords
        tex_verts = ("v2f", (-W, -W, W, -W, W, W, -W, W))
        tex_coords = ("t2f", (0.0, 0.0, W * 5.0, 0.0, W * 5.0, W * 5.0, 0.0, W * 5.0))
        self._layers.batch.add(4, gl.GL_QUADS, texture_group, tex_verts, tex_coords)

        for lane in self._lanes:
            lane.register(group=self._layers.road, batch=self._layers.batch)
        # setup objects
        for obj in self._objects:
            obj.register(group=self._layers.object, batch=self._layers.batch)

        # Setup text
        if not paper:
            self._setup_text(
                self._labels, group=self._texts.text, batch=self._texts.batch
            )
        # Setup Cars
        for car in self._cars:
            car.register(group=self._layers.car, batch=self._layers.batch)
        self._main_car.register(group=self._layers.main_car, batch=self._layers.batch)

    def _setup_text(self, labels, group, batch):
        assert self._window is not None
        labels["main"] = pyglet.text.Label(
            "Speed: ",
            font_name="Palatino",
            font_size=15,
            # y=300,
            x=30,
            y=self._window.height - 90,
            width=200,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            group=group,
            batch=batch,
        )
        labels["throttle"] = pyglet.text.Label(
            "Throttle: ",
            font_name="Palatino",
            font_size=15,
            x=30,
            y=self._window.height - 70,
            # y=300,
            width=200,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            group=group,
            batch=batch,
        )
        labels["brake"] = pyglet.text.Label(
            "Brake: ",
            font_name="Palatino",
            font_size=15,
            x=30,
            # y=300,
            y=self._window.height - 50,
            width=200,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            group=group,
            batch=batch,
        )
        labels["steer"] = pyglet.text.Label(
            "Steer: ",
            font_name="Palatino",
            font_size=15,
            x=30,
            y=self._window.height - 30,
            width=200,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            group=group,
            batch=batch,
        )
        labels["crash"] = pyglet.text.Label(
            "",
            font_name="Palatino",
            font_size=15,
            x=30,
            y=self._window.height - 110,
            width=200,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            group=group,
            batch=batch,
        )

    def _update_text(self, text=None):
        speed = self._main_car.state[0, 3]
        brake = self._main_car.brake
        throttle = self._main_car.throttle
        steer = self._main_car.steer

        state = self.state
        labels = self._labels
        const = self.constraints_fn(state[None, :], jnp.zeros((1, 1, 2)))
        crash = (
            onp.sum(const["offtrack"])
            + onp.sum(const["collision"]) if "collision" in const else 0
            + onp.sum(const["crash_objects"])
            if self._objects
            else 0
        ) > 0
        labels["main"].text = f"Speed: {speed * self._speed_factor:.2f} mph"
        labels["brake"].text = f"Brake: {brake:.2f}"
        labels["throttle"].text = f"Throttle: {throttle:.2f}"
        labels["steer"].text = f"Steer: {steer:.2f}"
        labels["crash"].text = f""
        labels["main"].color = (255, 255, 255, 255)
        labels["brake"].color = (255, 255, 255, 255)
        labels["throttle"].color = (255, 255, 255, 255)
        labels["steer"].color = (255, 255, 255, 255)
        labels["crash"].color = (255, 0, 0, 255)
        if brake > self._main_car.max_brake:
            labels["brake"].color = (255, 0, 0, 255)
        if steer > self._main_car.max_steer:
            labels["steer"].color = (255, 0, 0, 255)
        if throttle > self._main_car.max_throttle:
            labels["throttle"].color = (255, 0, 0, 255)
        if speed > self._main_car.max_speed:
            labels["main"].color = (255, 0, 0, 255)
        if text is not None:
            labels["main"].text += "\n" + text
        if crash:
            labels["crash"].text = "Crash!!"
        for key, val in labels.items():
            val.draw()

    def _set_viz_heat(self, weights):
        assert isinstance(weights, DictList)
        # Clean up weights input
        weights = weights.prepare(self.features_keys)

        def val(xs, ys, weights=weights):
            state = deepcopy(self.state)
            n_states = len(xs)
            cost_fn = partial(
                self._main_car.cost_runtime,
                weights=weights.repeat(n_states).numpy_array(),
            )
            states = jnp.repeat(state, n_states, axis=0)  # (n_states, xdim)
            phis = jnp.array([jnp.pi / 3] * n_states)
            main_idx = self._indices["main_car"]
            # states[:, main_idx[0] : main_idx[0] + 3] = jnp.stack([xs, ys, phis], axis=1)
            states = states.at[:, main_idx[0] : main_idx[0] + 3].set(
                jnp.stack([xs, ys, phis], axis=1)
            )
            acts = jnp.zeros((n_states, self.udim))
            return cost_fn(states, acts)

        self._viz_heat_fn = val

    def _set_viz_constraints(self, constraints_key):
        def val(xs, ys):
            constraint_fn = self._constraints_dict[constraints_key]
            n_states = len(xs)
            state = deepcopy(self.state)
            states = onp.repeat(state, n_states, axis=0)  # (n_states, xdim)
            phis = onp.array([onp.pi / 3] * n_states)
            main_idx = self._indices["main_car"]
            states[:, main_idx[0] : main_idx[0] + 3] = onp.stack([xs, ys, phis], axis=1)
            states = states[None, :, :]  # (1, n_states, xdim)
            acts = onp.zeros((1, n_states, self.udim))  # (1, n_states, udim)
            return 0.5 * constraint_fn(states, acts)

        self._viz_constraint_fn = val

    def _draw_heatmap(self, heat_fn):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm

        self._cm = matplotlib.cm.coolwarm

        center = self.main_car.state[0, :2]
        c0 = center - onp.array([1.0, 1.0]) / self._magnify
        c1 = center + onp.array([1.0, 1.0]) / self._magnify

        SIZE = (32, 32)
        # SIZE = (16, 16)
        # Sweep for cost values
        xs = jnp.linspace(c0[0], c1[0], SIZE[0])
        ys = jnp.linspace(c0[1], c1[1], SIZE[1])
        xvs, yvs = jnp.meshgrid(xs, ys)
        vals = heat_fn(xvs.flatten(), yvs.flatten())
        vals = vals.reshape(SIZE)
        vals = (vals - onp.min(vals)) / (onp.max(vals) - onp.min(vals) + 1e-6)
        # Convert to color map and draw
        vals = self._cm(vals)  # (SIZE[0], SIZE[1], 4)
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
