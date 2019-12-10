"""Acquisition Procedures.

Given a new task, compute a saliency score for the new task.

Include:
    * Info Gain: H(X) - H(X | Y)
    * Max discrepancy (ratio): f(X, Y) / f(X)
"""


# infogain (fn): information gain (-entropy) in p(w* | obs)


class ActiveInfoGain(object):
    """
    Args:
        model (object): IRD Model
    """

    def __init__(self, env, model):
        self._env = env
        self._model = model
        self._tasks = []

    def __call__(self, task_name, task):
        """
        Pseudocode:
        ```
        self._tasks.append(task)
        curr_sample_ws, curr_sample_feats = self._model.get_samples(task_name)
        user_w, user_feat = random.choice(curr_sample_ws, curr_sample_feats)
        # Collect featurs on new task
        task_feats = self._model.collect_feats(curr_sample_ws, task)
        for feats in task_feats:
            new_log_prob = user_w.dot(feats)
        new_sample_ws = resample(curr_sample_ws, new_log_prob + log_prob)
        return entropy(curr_sample_ws) - entropy(new_sample_ws)
        ```
        """
        pass


class ActiveRatioTest(ActiveInfoGain):
    def __init__(self, env, model, method="max"):
        super().__init__(env, model)
        self._method = method

    def __call__(self, task_name, task):
        """
        Pseudocode:
            self._tasks.append(task)
            curr_samples = self._model.get_samples(task_name)
            for s in curr_samples:
                new_log_prob = w.dot(feats)
            return np.mean(new_log_prob - log_prob)
        """
        pass
