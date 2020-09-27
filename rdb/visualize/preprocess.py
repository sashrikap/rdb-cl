import numpy as onp
from collections import OrderedDict
from copy import deepcopy

"""
Simple Data Preprocessing functions
"""


def normalize_features(features, eps=1e-8):
    """
    Normalize features across multiple runs

    Param
    : features : dict features across different tasks
                 features[keys] = (#task1, #task2, ....)
    """
    assert isinstance(features, dict), "Wrong feature type provided"

    info = OrderedDict({"mean": {}, "std": {}})
    if len(features) == 0:
        return features, info

    normalized_features = deepcopy(features)
    for key in features.keys():
        mean = onp.mean(features[key])
        std = onp.std(features[key])
        info["std"][key] = std
        info["mean"][key] = mean
        features[key] = (features[key] - mean) / (std + eps)
    return features, info
