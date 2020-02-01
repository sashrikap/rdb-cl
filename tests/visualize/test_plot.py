from rdb.visualize.plot import *
import numpy as np


def get_weights(num_feats=5, num_weights=1000):
    output = []
    for i in range(num_weights):
        w = {}
        for fi in range(num_feats):
            w[str(fi)] = 10 * (np.random.random() - 0.5)
        output.append(w)
    return output


def test_weights():
    weights = get_weights()
    plot_weights(
        weights_dicts=weights,
        highlight_dicts=weights[:5],
        highlight_colors=["r", "b", "k", "g", "c"],
        highlight_labels=["1", "2", "3", "4", "5"],
        path="data/test/test_plot_weights.png",
        log_scale=False,
        title="Test plot",
    )


def test_weight_comparisons():
    all_weights = [get_weights(), get_weights(), get_weights()]
    plot_weights_comparison(
        all_weights_dicts=all_weights,
        all_weights_colors=["b", "g", "r"],
        path="data/test/test_plot_compare.png",
        log_scale=False,
        title="Test plot",
    )
