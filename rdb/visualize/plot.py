import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import rdb
from imageio import imread
from os.path import join, dirname, isfile, isdir
from os import listdir
from matplotlib import cm, gridspec
from mpl_toolkits import mplot3d
from IPython.display import display, Video
from IPython.display import display, update_display
from ipywidgets import (
    interact,
    interactive,
    fixed,
    Button,
    Layout,
    interact_manual,
    widgets,
)

"""
Plotting Utilities
"""


def plot_3d(xs, ys, zs, xlabel="", ylabel="", title=""):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    X, Y = np.meshgrid(xs, ys)
    Z = np.array(zs)
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


def plot_episode(frames, name="Figure"):
    nframes = len(frames)
    fig, axes = plt.subplots(num=name)
    axes.axis("off")

    def view_image(i):
        axes.imshow(frames[i])

    # Interactive widgets
    player = widgets.Play(
        interval=100,
        value=0,
        min=0,
        max=100,
        step=1,
        description="Press play",
        disabled=False,
    )
    slider = widgets.IntSlider(min=0, max=nframes - 1, layout=Layout(width="3000px"))
    widgets.jslink((player, "value"), (slider, "value"))
    interactive(view_image, i=slider)
    box = widgets.HBox([player, slider])
    display(box)
    # Display image
    slider.value = 0
    axes.imshow(frames[0])


marker_points = []
record_images = []
video_display = None
display_vx, display_vy = 0.0, 0.0


def plot_3d_interactive_features(
    xs,
    ys,
    zs,
    feats,
    video_path_fn=None,
    thumbnail_path_fn=None,
    video_width=300,
    step=0.1,
):
    # transparency
    ALPHA = 0.7
    # Plot 3D
    fig = plt.figure(figsize=(9, 6))
    gs0 = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax0 = fig.add_subplot(gs0[:, :-1], projection="3d")
    ax0.clear()
    X3d, Y3d = np.meshgrid(xs, ys)
    Z3d = np.array(zs)
    surf = ax0.plot_surface(
        X3d,
        Y3d,
        Z3d,
        cmap=cm.coolwarm,
        alpha=ALPHA,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax0.set_xlabel("x value")
    ax0.set_ylabel("y value")

    # Plot assisting figures
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])
    ax10 = fig.add_subplot(gs01[:-1, -1])
    ax11 = fig.add_subplot(gs01[-1, -1])
    ax10.clear()
    ax11.clear()
    ax11.axis("off")
    plt.tight_layout()

    # Feature range
    feat_keys = list(feats.keys())
    min_feats = [np.min(feats[key]) for key in feat_keys]
    max_feats = [np.max(feats[key]) for key in feat_keys]

    ix, iy = 0, 0
    vals = [feats[key][ix][iy] for key in feat_keys]
    xbar = np.arange(len(feat_keys))
    ax10.clear()
    ax10.set_xticks(xbar, feat_keys)
    ax10.set_ylim(min(min_feats), max(max_feats))
    ax10.bar(xbar, vals)

    def onclick(event):
        global ex, ey
        ex, ey = event.xdata, event.ydata
        # plt.suptitle(f"Clicked at x({ex:.2f}), y({ey:.2f})")
        # ix = np.argmin(np.abs(xs - ex))
        # iy = np.argmin(np.abs(ys - ey))
        # plt.suptitle(f"x({ex:.2f}) y({ey:.2f}) rew({rews[]})")
        # plot_feats(ix, iy)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # Create reference anchor
    def display_video(video_path=None):
        pass

    def display_thumbnail(thumbnail_path=None):
        pass

    if thumbnail_path_fn is None:
        thumbnail_path_fn = lambda x, y: None
    if video_path_fn is None:
        video_path_fn = lambda x, y: None
    points = []

    def plot_feats(ix, iy):
        global marker_points
        global display_vx, display_vy
        # plot bar chart
        OFFSET = 4
        SIZE = 40

        vals = [feats[key][ix][iy] for key in feat_keys]
        xbar = np.arange(len(feat_keys))
        ax10.clear()
        ax10.set_ylim(min(min_feats), max(max_feats))
        ax10.bar(xbar, vals, tick_label=feat_keys)
        ax10.set_xticklabels(ax10.get_xticklabels(), rotation=15, ha="right")
        # plot 3d surface marker
        vx, vy = xs[ix], ys[iy]
        display_vx, display_vy = vx, vy
        vz = zs[ix][iy]
        for p in marker_points:
            p.remove()
        marker_points = []
        for x, y, z in zip([vx, vx], [vy, vy], [vz + OFFSET, vz + OFFSET]):
            point = ax0.scatter(x, y, z, marker="^", c="g", s=SIZE)
            marker_points.append(point)
        # plot title
        plt.suptitle(f"x({xs[ix]:.2f}) y({ys[iy]:.2f}) rew({zs[ix][iy]:.3f})")

        # show video (optional)
        # video_path = video_path_fn(ix, iy)
        thumbnail_path = thumbnail_path_fn(vx, vy)
        video_path = video_path_fn(vx, vy)
        if not thumbnail_path:
            ax11.clear()
        ax11.axis("off")
        display_video(video_path)
        display_thumbnail(thumbnail_path)

    plot_feats(0, 0)

    def slide_xy(x, y):
        global ix, iy
        ix = int(np.argmin(np.abs(xs - x)))
        iy = int(np.argmin(np.abs(ys - y)))
        plot_feats(ix, iy)

    # slider widgets
    slider_x = widgets.FloatSlider(
        min=min(xs),
        max=max(xs) + 0.01,
        value=0.0,
        step=step,
        layout=Layout(width="800px"),
    )
    slider_y = widgets.FloatSlider(
        min=min(ys),
        max=max(ys) + 0.01,
        value=0.0,
        step=step,
        layout=Layout(width="800px"),
    )
    interactive(slide_xy, x=slider_x, y=slider_y)

    video_player = widgets.Play(
        interval=100,
        value=0,
        min=0,
        max=100,
        step=1,
        description="Press play",
        disabled=False,
    )
    # video_slider = widgets.IntSlider()
    # widgets.jslink((video_player, "value"), (video_slider, "value"))

    def display_video(video_path=None):
        """
        global record_images
        record_images = []
        if video_path is None:
            video_player.max = 0
        else:
            files = sorted([f for f in listdir(video_path) if ".png" in f])
            nframes = len(files)
            record_images = [imread(join(video_path, f)) for f in files]
            video_slider.value = 0
            ax11.imshow(record_images[0])
        """
        global video_display
        if video_path is not None:
            if video_display is None:
                video = Video(video_path, width=video_width)
                video_display = display(video, display_id=True).display_id
            else:
                video = Video(video_path, width=video_width)
                update_display(video, display_id=video_display)

    def display_thumbnail(thumbnail_path=None):
        if thumbnail_path is not None:
            ax11.imshow(imread(thumbnail_path))

    play_button = widgets.Button(description="Play MP4")
    """def on_button_clicked(_):
        global display_vx, display_vy
        video_path = video_path_fn(display_vx, display_vy)
        display_video(video_path)
        #with out:
        #print('Something happens!')
    play_button.on_click(on_button_clicked)"""

    # def view_image(i):
    #    ax11.imshow(record_images[i])

    # interactive(view_image, i=video_slider)
    # ax11.imshow(record_images[0])

    # selector_box = widgets.VBox([slider_x, slider_y, video_display])
    selector_box = widgets.VBox([slider_x, slider_y])
    display(selector_box)
    # video_box = widgets.HBox([video_player, video_slider])
    # ui = widgets.HBox([selector_box, play_button])
    # display(ui)
    global video_display
    video_display = None
    display_video(video_path_fn(0.0, 0.0))
    display_thumbnail(thumbnail_path_fn(0.0, 0.0))
    display_thumbnail(thumbnail_path_fn(0.0, 0.5))
