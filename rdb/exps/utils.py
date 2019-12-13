import matplotlib.pyplot as plt
import jax.numpy as np

PLOT_BINS = 100
MAX_WEIGHT = 8.0


def plot_samples(samples_dicts, highlight_dict=None, save_path=None):
    plt.figure(figsize=(8, 6), dpi=80)
    n_values = len(samples_dicts[0].values())
    for i, key in enumerate(samples_dicts[0].keys()):
        values = [np.log(s[key]) for s in samples_dicts]
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
            val = highlight_dict[key]
            bin_i = np.argmin(np.abs(bins[:-1] - val))
            patches[bin_i].set_fc("r")
        plt.title(key)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
