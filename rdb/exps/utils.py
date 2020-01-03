import matplotlib.pyplot as plt
import jax.numpy as np

PLOT_BINS = 100
MAX_WEIGHT = 8.0


def plot_weights(weights_dicts, highlight_dict=None, path=None):
    fig = plt.figure(figsize=(8, 6), dpi=80)
    n_values = len(weights_dicts[0].values())
    for i, key in enumerate(weights_dicts[0].keys()):
        values = [np.log(s[key]) for s in weights_dicts]
        plt.subplot(n_values, 1, i + 1)
        n, bins, patches = plt.hist(
            values,
            PLOT_BINS,
            range=(-MAX_WEIGHT, MAX_WEIGHT),
            density=True,
            facecolor="b",
            alpha=0.75,
        )
        ## Highlight value
        if highlight_dict is not None:
            # # not good, somethings bin too low and submerged
            # val = highlight_dict[key]
            # bin_i = np.argmin(np.abs(bins[:-1] - val))
            # patches[bin_i].set_fc("r")
            val = highlight_dict[key]
            plt.axvline(x=np.log(val), c="r")
        plt.title(key)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
