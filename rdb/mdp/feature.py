import jax.numpy as np


class Feature(object):
    """ General state-based feature

    Usage:
    feat = f(state)
    feat: scalar
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, state):
        state = np.array(state)
        return self.f(state)


class FeatureList(object):
    """ List of driving relevant features """

    def __init__(self):
        self._feats = []
        self._weights = []

    def add_feature(self, feat, weight):
        assert type(feat) == Feature, "Must be a feature"
        self._feats.append(feat)
        self._weights.append(weight)

    def __call__(self, state):
        feats = []
        for f in zip(self._feats):
            feats.append(f(state))
        return np.array(feats)

    def rewards(self, state):
        rews = []
        for f, w in zip(self._feats, self._weights):
            rews.append(f(state) * w)
        return np.array(rews)
