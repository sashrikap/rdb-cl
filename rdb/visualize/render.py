from moviepy.editor import ImageSequenceClip
from functools import partial
from matplotlib import pyplot as plt
from os import makedirs
from os.path import join
from scipy.misc import imsave, imresize
import multiprocessing


def save_video(frames, fps, width, path):
    clip = ImageSequenceClip(frames, fps=fps).resize(width=width)
    clip.write_videofile(path, logger=None)


def forward_env(env, actions, init_state=None):
    # Render and save environment given pre-specified actions
    env.reset()
    if init_state is not None:
        env.state = init_state
    frames = []
    subframe_op = getattr(env, "sub_render", None)
    has_subframe = callable(subframe_op)

    for act in actions:
        env.step(act)
        if has_subframe:
            for s_i in range(env.subframes):
                frames.append(env.sub_render("rgb_array", s_i))
        else:
            frames.append(env.render("rgb_array"))
    return frames


def render_env(
    env, state, actions, fps, path="data/video.mp4", width=450, savepng=False
):
    frames = forward_env(env, actions, state)
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
