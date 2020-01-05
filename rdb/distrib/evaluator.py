"""Distributed evaluator class. Used for parallel evaluation.
"""
import ray


class Evaluator(object):
    def __init__(self):
        self._eval_result = None

    def evaluate(self):
        pass


@ray.remote
class EvaluatorParticle(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self, particles, target_w):
        self._eval_result = particles.compare_with(target_w)

    def get_result(self):
        return self._eval_result
