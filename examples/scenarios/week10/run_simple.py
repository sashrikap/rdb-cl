import gym
import jax
import copy
import jax.numpy as jnp
import numpy as onp
from numpngw import write_apng
from sys import platform
import rdb.envs.drive2d

from time import time, sleep
from tqdm import tqdm
from rdb.optim.mpc import build_mpc
from rdb.optim.runner import Runner
from rdb.infer.dictlist import DictList
from PIL import Image


ENV_NAME = "Week10_01"
TASK = (0, 0)
HORIZON = 10

HEATMAP_WEIGHTS = {
    "dist_cars": 5,
    "dist_lanes": 5,
    "dist_fences": 0.35,
    "speed": 5,
    "control": 0.1,
}


def render_popup():
    env = gym.make(ENV_NAME)
    env.set_task(TASK)
    obs = env.reset()
    # Display rendering to a pop-up window
    env.render("human")

    done = False
    t = 0
    while t < HORIZON:
        obs, rew, done, truncated, info = env.step(env.action_space.sample())
        t += 1
        print("Rendering pop-up mode step {}/{}".format(t, HORIZON))
        env.render("human")
        sleep(0.1)


def render_raw_fps_to_file(draw_heat=False):
    env = gym.make(ENV_NAME)
    horizon = 10
    env.set_task(TASK)
    obs = env.reset()

    # Script that ensures pyglet renders -> virtual display -> file
    if platform == "win32": # Windows
        from pyvirtualdisplay import Display

        display = Display(visible=0, size=(1400, 900))
        display.start()

    # Save rendering to a file
    env.render("rgb_array")
    done = False
    t = 0
    frames = []

    def _resize(img, ratio):
        import cv2

        old_w, old_h = img.shape[1], img.shape[0]
        new_w = int(old_w * ratio)
        new_h = int(old_h * ratio)
        img = cv2.resize(img.astype(onp.uint8), (new_w, new_h))
        return img

    while t < HORIZON:
        obs, rew, done, truncated, info = env.step(env.action_space.sample())
        t += 1
        print(
            "Rendering raw fps mode heatmap {} step {}/{}".format(draw_heat, t, HORIZON)
        )
        img = env.render(
            "rgb_array",
            draw_heat=draw_heat,
            weights=DictList([HEATMAP_WEIGHTS], jax=True),
        )
        frames.append(_resize(img, 0.4))

    # Specify your path & filename
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save("render-raw_fps-heatmap-{}.gif".format(draw_heat), save_all = True, append_images=imgs[1:], duration=100, loop=0)


def render_high_fps_to_file():
    """This mode leverages pyglet to interpolate between frames, which renders higher-fps videos
    """
    from rdb.visualize.render import forward_env

    env = gym.make(ENV_NAME)
    horizon = 10
    env.set_task(TASK)
    obs = env.reset()
    actions = onp.array([env.action_space.sample() for _ in range(horizon)])[
        None
    ]  # (nenv, T, -1)
    init_state = env.state  # (nenv, -1)
    print("Rendering high fps mode")

    # This handles (1) env reset. (2) env forwarding, (3) frame collection
    frames = forward_env(env, actions, init_state)
    delay = int(100 * horizon / len(frames))

    def _resize(img, ratio):
        import cv2

        old_w, old_h = img.shape[1], img.shape[0]
        new_w = int(old_w * ratio)
        new_h = int(old_h * ratio)
        img = cv2.resize(img.astype(onp.uint8), (new_w, new_h))
        return img

    frames = [_resize(img, 0.4) for img in frames]
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save("render-high-fps.gif", save_all = True, append_images=imgs[1:], duration=100, loop=0)

    print("Rendering done")


def main():
    render_popup()
    render_raw_fps_to_file(draw_heat=False)
    render_raw_fps_to_file(draw_heat=True)
    render_high_fps_to_file()


if __name__ == "__main__":
    render_mode = ""
    main()
