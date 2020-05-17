from rdb.visualize.plot import *
from rdb.infer import DictList
import numpy as np


def get_weights(num_feats=5, num_weights=1000):
    output = []
    for i in range(num_weights):
        w = {}
        for fi in range(num_feats):
            w[str(fi)] = 10 * (np.random.random() - 0.5)
        output.append(w)
    return DictList(output)


def ttest_weights():
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


def ttest_weight_comparisons():
    all_weights = [get_weights(), get_weights(), get_weights()]
    plot_weights_comparison(
        all_weights_dicts=all_weights,
        all_weights_colors=["b", "g", "r"],
        all_labels=["a", "b", "c"],
        path="data/test/test_plot_compare.png",
        log_scale=False,
        title="Test plot",
    )


def ttest_weight_correlations():
    N = 10
    all_scores = [np.random.random(N), np.random.random(N)]
    all_labels = ["Abc", "Def"]
    plot_ranking_corrs(
        all_scores,
        all_labels,
        path="data/test/test_plot_correlation.png",
        title="Abc vs Def",
    )


def test_weight_2d():
    chain = get_weights()
    plot_weights_2d(
        weights_dicts=chain,
        # weights_colors=["b", "g"],
        keys=chain.keys(),
        path="data/test/test_plot_2d.png",
        log_scale=False,
        title="Test plot",
    )
