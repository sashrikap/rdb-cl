import gym
import copy
import numpy as onp
from functools import partial
from matplotlib import pyplot as plt
from os import makedirs
from os.path import join
from scipy.misc import imsave, imresize
from rdb.exps.utils import Profiler
import multiprocessing


class RenderEnv(gym.Env):
    """Environment with interpolation-based subrender.

    Workflow:
        env.prepare_subrender()
        for t in range(T):
            env.step(acs)
            for i in range(subframes):
                env.step_subrender()

    """

    def __init__(self, subframes):
        # super().__init__()
        self._subframes = subframes
        self._subcount = 0
        self._state, self._prev_state, self._curr_state = None, None, None

    @property
    def subframes(self):
        return self._subframes

    def step_subrender(self, *args, **kwargs):
        self._prev_state = copy.deepcopy(self._curr_state)
        self.state = self._curr_state
        self._subcount = 0
        self.step(*args, **kwargs)
        self._curr_state = copy.deepcopy(self.state)

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_subrender(self):
        self._prev_state = copy.deepcopy(self.state)
        self._curr_state = copy.deepcopy(self.state)
        self._subcount = 0

    def subrender(self, *args, **kwargs):
        ratio = float(self._subcount / self._subframes)
        sub_state = self._prev_state * (1 - ratio) + self._curr_state * ratio
        self._subcount += 1
        self.state = sub_state
        return self.render(*args, **kwargs)

    def render(self, mode, text="", *args, **kwargs):
        pass


def save_video(frames, fps, width, path):
    ## Todo: this seems to be at odds with gcp headless servers
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=fps).resize(width=width)
    clip.write_videofile(path, logger=None)


def forward_env(env, actions, init_state=None, text=None):
    # Render and save environment given pre-specified actions
    env.state = init_state
    actions = onp.array(actions)
    frames = []
    has_subframe = env.subframes > 0

    nacs = actions.shape[1]
    if has_subframe:
        env.prepare_subrender()
        for ai in range(nacs):
            env.step_subrender(actions[:, ai])
            for s_i in range(env.subframes):
                frames.append(env.subrender(mode="rgb_array", text=text))
    else:
        for ai in range(nacs):
            env.step(actions[:, ai])
            frames.append(env.render(mode="rgb_array", text=text))

    # frames = frames[1:]
    return frames


def render_env(
    env, state, actions, fps, path="data/video.mp4", width=450, savepng=False, text=None
):
    frames = forward_env(env, actions, state, text=text)
    save_video(frames, int(fps / env.dt), width, path)
    if savepng:
        dirname = path.replace(".mp4", "")
        makedirs(dirname, exist_ok=True)
        for i, frame in enumerate(frames):
            frame = imresize(frame, (width, width))
            imsave(join(dirname, f"frame_{i:03d}.png"), frame)


def batch_render_env(env, states, actions, paths, fps, width, workers=4):
    all_frames = []
    for state, acts, path in zip(states, actions, paths):
        all_frames.append(forward_env(env, acts, state))

    with multiprocessing.Pool(processes=workers) as pool:
        results = []
        for frames, path in zip(all_frames, paths):
            results.append(
                pool.apply_async(
                    partial(save_video, frames=frames, fps=fps, width=width, path=path)
                )
            )
        [r.get() for r in results]
