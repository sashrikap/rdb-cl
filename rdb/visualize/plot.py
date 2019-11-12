import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ipywidgets as widgets

from matplotlib import cm
from mpl_toolkits import mplot3d
from ipywidgets import interact, interactive, fixed
from IPython.display import display

"""
Plotter Utilities
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
    slider = widgets.IntSlider(min=0, max=nframes - 1)
    widgets.jslink((player, "value"), (slider, "value"))
    interactive(view_image, i=slider)
    box = widgets.HBox([player, slider])
    display(box)
    # Display image
    slider.value = 0
    axes.imshow(frames[0])


def plot_3d_interactive_features(xs, ys, zs, feats):
    """ Visualize trajectory features in an interactive way

    Note
    [1] Environment has 2 DoF
    [2] Only works on Jupyter Notebook
    """
    assert (
        matplotlib.get_backend() == "nbAgg"
    ), "Interactive features only work in notebook mode"
