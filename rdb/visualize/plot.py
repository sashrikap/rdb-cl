import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as onp
import rdb
from rdb.exps.utils import Profiler
from imageio import imread
from os.path import join, dirname, isfile, isdir
from os import listdir
from matplotlib import cm, gridspec
from IPython import get_ipython

# from mpl_toolkits import mplot3d

if get_ipython() is not None:
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


def plot_weights(
    weights_dicts,
    highlight_dicts=[],
    highlight_colors=[],
    highlight_labels=[],
    path=None,
    title=None,
    max_weights=8.0,
    bins=100,
    hist_weights=None,
    log_scale=True,
    viz_normalized_key=None,
    **kwargs,
):
    """Plot weights for visualizing.

    Args:
        highlight_dicts (list): list of weight dictionaries
        highlight_colors (list): list of colors for highlighting these weights
        highlight_labels (list): list of labels for denoting these weights
        max_weights (float): log range of weights ~ (-max_weights, max_weights)
        hist_weights (list): plotting weights associated with each particle

    """
    assert len(highlight_dicts) == len(highlight_colors) == len(highlight_labels)

    n_values = len(weights_dicts[0].values())
    fig, axs = plt.subplots(n_values, 1, figsize=(20, 2 * n_values), dpi=80)
    weights_dicts = weights_dicts.log() if not log_scale else weights_dicts
    for i, key in enumerate(sorted(list(weights_dicts[0].keys()))):
        values = weights_dicts[key]
        if viz_normalized_key is not None and viz_normalized_key in weights_dicts:
            values /= weights_dicts[viz_normalized_key]
        n, bins, patches = axs[i].hist(
            values,
            bins,
            histtype="stepfilled",
            range=(-max_weights, max_weights),
            density=True,
            facecolor="b",
            ec="k",
            alpha=0.3,
            weights=hist_weights,
        )
        ybottom, ytop = axs[i].get_ylim()
        gap = (ytop - ybottom) / (len(highlight_dicts) - 1 + 1e-8)
        ## Highlight value
        for j, (d, c, label) in enumerate(
            zip(highlight_dicts, highlight_colors, highlight_labels)
        ):
            if d is None or key not in d:
                continue
            val = d[key]
            if not log_scale:
                val = onp.log(val)
            axs[i].axvline(x=val, c=c)
            # add 0.05 gap so that labels don't overlap
            axs[i].text(val, ybottom + gap * j, label, size=10)
        axs[i].set_xlabel(key)
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        plt.savefig(path)
        plt.close("all")
    else:
        plt.show()


def plot_weights_2d(
    weights_dicts,
    # weights_color,
    keys,
    highlight_dicts=[],
    highlight_colors=[],
    highlight_labels=[],
    path=None,
    title=None,
    max_weights=8.0,
    log_scale=True,
    loc="upper right",
    **kwargs,
):
    """Plot weights for visualizing.

    Args:
        weight_dicts (DictList)
        key_i (str): first feature key
        key_j (str): second feature key
        highlight_dicts (list): list of weight dictionaries
        highlight_colors (list): list of colors for highlighting these weights
        highlight_labels (list): list of labels for denoting these weights
        max_weights (float): log range of weights ~ (-max_weights, max_weights)

    """
    assert len(highlight_dicts) == len(highlight_colors) == len(highlight_labels)
    # fig, axs = plt.subplots(1, figsize=(8, 8), dpi=80)
    nkeys = len(keys)
    # For higher quality, choose dpi>=80
    fig, axs = plt.subplots(nkeys, nkeys, figsize=(6 * nkeys, 6 * nkeys), dpi=30)

    weights_dicts = weights_dicts.log() if not log_scale else weights_dicts

    for i, key_i in enumerate(keys):  # row
        for j, key_j in enumerate(keys):  # col
            if i == nkeys - 1:
                axs[i, j].set_xlabel(key_j, fontsize=18)
            if j == 0:
                axs[i, j].set_ylabel(key_i, fontsize=18)

            if i == j:
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xticklabels([])
                continue

            values_i = weights_dicts[key_i]
            values_j = weights_dicts[key_j]
            axs[i, j].plot(
                values_j,
                values_i,
                "^",
                # color=weights_color,
                # facecolor=color,
                markersize=3,
                alpha=0.5,
            )
            axs[i, j].set_xlim(-max_weights, max_weights)
            axs[i, j].set_ylim(-max_weights, max_weights)
            ## Highlight value
            for j, (d, c, label) in enumerate(
                zip(highlight_dicts, highlight_colors, highlight_labels)
            ):
                if d is None or key not in d:
                    continue
                val_i = d[key_i]
                val_j = d[key_j]
                if not log_scale:
                    val_i = onp.log(val_i)
                    val_j = onp.log(val_j)
                axs[i, j].plot(val_i, val_j, c=c)
                # add 0.05 gap so that labels don't overlap
                axs[i, j].text(val_i, val_j, label, size=10)
            # axs.legend(loc=loc)
            # axs[i, j].set_title(f"{key_i} vs {key_j}")
            axs[i, j].set_yticklabels([])
            axs[i, j].set_xticklabels([])
            # axs[i, j].set_xlabel(key_i)
            # axs[i, j].set_ylabel(key_j)

    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        plt.savefig(f"{path}.png")
        plt.close("all")
    else:
        plt.show()


def plot_weights_hist(
    all_weights_dicts,
    all_weights_colors,
    all_labels,
    path=None,
    title=None,
    max_weights=8.0,
    bins=100,
    log_scale=True,
    hist_weights=None,
    loc="upper right",
    **kwargs,
):
    """Plot different lists weights for visualizing (e.g. MCMC convergence).

    Args:
        all_weights_dicts (list(DictList)): list of weight dictionaries
            shape: list(nfeats * (nsamples,))
        highlight_colors (list): list of colors for highlighting these weights
        all_labels (list):
        highlight_labels (list): list of labels for denoting these weights
        max_weights (float): log range of weights ~ (-max_weights, max_weights)

    """

    assert len(all_weights_dicts) == len(all_weights_colors) == len(all_labels)
    axs = None
    # For each chain
    for weights_dicts, color, label in zip(
        all_weights_dicts, all_weights_colors, all_labels
    ):
        if len(weights_dicts) == 0:
            continue

        n_values = len(list(weights_dicts.keys()))
        if axs is None:
            fig, axs = plt.subplots(n_values, 1, figsize=(20, 2 * n_values), dpi=80)
        # For each key in all dicts of that chain
        for i, key in enumerate(sorted(list(weights_dicts.keys()))):
            n_weights = len(weights_dicts)
            values = onp.array(list(weights_dicts[key]))
            if not log_scale:
                values = onp.log(values)
            n, _, patches = axs[i].hist(
                values,
                bins,
                histtype="stepfilled",  # no vertical border
                range=(-max_weights, max_weights),
                density=True,
                label=label,
                facecolor=color,
                ec="k",  # border
                weights=hist_weights,
                alpha=0.25,
            )
            axs[i].set_xlabel(key)
    axs[0].legend(loc=loc)
    plt.tight_layout()
    if title is not None:
        fig.suptitle(title)
    if path is not None:
        plt.savefig(path)
        plt.close("all")
    else:
        plt.show()


def plot_rankings(
    main_val,
    main_label,
    auxiliary_vals=[],
    auxiliary_labels=[],
    path=None,
    title=None,
    yrange=None,
    loc="lower right",
    normalize=False,
    delta=0.0,
    annotate_rankings=False,
    figsize=(20, 10),
):
    """Plot main_val and auxiliary_vals based on ranking of main_val.

    Args:
        main_val (list): main value to be plotted
        main_label (list):
        auxiliary_values (list([])): >=0 list of other values
        auxiliary_labels (list([])): >=0 other labels
        path (str): save path
        title (str)
        loc (str): legend location
        normalize (bool):
        delta (float): avoid overlapping
    """
    idxs = onp.argsort(onp.array(main_val))
    _, ax = plt.subplots(figsize=figsize)
    all_vals = auxiliary_vals + [main_val]
    all_labels = auxiliary_labels + [main_label]
    # Sort based on keys (so that colors stay the same across plots)
    label_idxs = onp.argsort(all_labels)
    all_labels = [all_labels[i] for i in label_idxs]
    all_vals = [all_vals[i] for i in label_idxs]
    # Plot
    n_delta = 0.0
    eps = 1e-8
    for val, label in zip(all_vals, all_labels):
        val = onp.array(val, dtype=float)[idxs]
        if normalize:
            val = (val - onp.min(val)) / (onp.max(val) - onp.min(val) + eps)
        val += n_delta * delta  # avoid overlapping
        n_delta += 1
        xs = list(range(len(val)))
        ax.plot(xs, val, "^", label=label)
        if annotate_rankings:
            for i, xy in enumerate(zip(xs, val)):
                # ax.annotate(idxs[i], xy=xy, textcoords='data')
                ax.text(xy[0], xy[1], idxs[i], size=7)
    ax.legend(loc=loc)
    if title is not None:
        ax.set_title(title)
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])
    if path is not None:
        plt.savefig(path)
        plt.close("all")
    else:
        plt.show()


def plot_ranking_corrs(
    values,
    labels,
    path=None,
    title=None,
    yrange=None,
    delta=0.01,
    annotate_identites=True,
    figsize=(10, 10),
):
    """Plot cross correlations between pairs of values.

    Args:
        values (list): main value to be plotted
        labels (list):
        path (str): save path
        title (str)
        delta (float): avoid overlapping

    """
    assert len(values) == 2
    vals_i, vals_j = values[0], values[1]
    label_i, label_j = labels[0], labels[1]
    # normalize:
    eps = 1e-9
    diff_i = onp.max(vals_i) - onp.min(vals_i) + eps
    diff_j = onp.max(vals_j) - onp.min(vals_j) + eps
    vals_i = (vals_i - onp.min(vals_i)) / diff_i
    vals_j = (vals_j - onp.min(vals_j)) / diff_j

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(vals_i, vals_j, "^")
    if annotate_identites:
        for idx, (vi, vj) in enumerate(zip(vals_i, vals_j)):
            ax.text(vi + delta, vj, idx, size=12)
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1)
    ax.set_xlabel(label_i)
    ax.set_ylabel(label_j)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title is not None:
        ax.set_title(title)
    if yrange is not None:
        ax.set_ylim(yrange[0], yrange[1])
    if path is not None:
        plt.savefig(path)
        plt.close("all")
    else:
        plt.show()


def plot_3d(xs, ys, zs, xlabel="", ylabel="", title="", figsize=(20, 10)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")
    X, Y = onp.meshgrid(xs, ys)
    Z = onp.array(zs)
    surf = ax.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


def plot_episode(frames, name="Figure", figsize=(20, 10)):
    nframes = len(frames)
    fig, axes = plt.subplots(num=name, figsize=figsize)
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
    X3d, Y3d = onp.meshgrid(xs, ys)
    Z3d = onp.array(zs)
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
    min_feats = [onp.min(feats[key]) for key in feat_keys]
    max_feats = [onp.max(feats[key]) for key in feat_keys]

    ix, iy = 0, 0
    vals = [feats[key][ix][iy] for key in feat_keys]
    xbar = onp.arange(len(feat_keys))
    ax10.clear()
    ax10.set_xticks(xbar, feat_keys)
    ax10.set_ylim(min(min_feats), max(max_feats))
    ax10.bar(xbar, vals)

    def onclick(event):
        global ex, ey
        ex, ey = event.xdata, event.ydata
        # plt.suptitle(f"Clicked at x({ex:.2f}), y({ey:.2f})")
        # ix = onp.argmin(onp.abs(xs - ex))
        # iy = onp.argmin(onp.abs(ys - ey))
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
        xbar = onp.arange(len(feat_keys))
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
        ix = int(onp.argmin(onp.abs(xs - x)))
        iy = int(onp.argmin(onp.abs(ys - y)))
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
